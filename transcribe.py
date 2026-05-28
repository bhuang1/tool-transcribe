#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-whisper"]
# ///
"""Record from the mic until ENTER, transcribe locally with mlx-whisper, copy to clipboard.

Python port of transcribe.sh. Where the bash version shells out to a whisper.cpp binary,
this uses mlx-whisper, which runs on Apple's MLX framework (Metal GPU on Apple Silicon).
Models are pulled automatically from the `mlx-community` org on Hugging Face, so there's
no C++ build step and no manual model files to manage.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_AUDIO_FILE = SCRIPT_DIR / "temp_audio.wav"

# Virtual / non-physical devices to skip when auto-picking a microphone.
VIRTUAL_DEVICES = ("ZoomAudioDevice", "Microsoft Teams", "BlackHole", "Soundflower", "virtual")

# whisper.cpp quantization suffixes to strip when mapping to an mlx-community repo.
QUANT_SUFFIXES = ("-q8_0", "-q5_1", "-q5_0", "-q4_1", "-q4_0", "-q4_k", "-q5_k")


def read_env_var(name: str) -> str:
    """Read a variable from env.sh (sourced in a subshell so comments are handled)."""
    env_sh = SCRIPT_DIR / "env.sh"
    if env_sh.exists():
        result = subprocess.run(
            ["bash", "-c", f'source "{env_sh}" && printf "%s" "${{{name}}}"'],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    return ""


def resolve_mlx_repo() -> str:
    """Resolve the model to pass to mlx-whisper (an mlx-community repo or a local path).

    MLX_WHISPER_MODEL overrides everything (a HF repo or a local model folder); it may be
    set inline in the environment or in env.sh. Otherwise the whisper.cpp WHISPER_MODEL is
    mapped to an mlx-community repo, e.g.
    'ggml-large-v3-turbo-q8_0' -> 'mlx-community/whisper-large-v3-turbo'.
    """
    override = os.environ.get("MLX_WHISPER_MODEL") or read_env_var("MLX_WHISPER_MODEL")
    if override:
        return override

    name = read_env_var("WHISPER_MODEL") or "ggml-large-v3-turbo"
    if name.startswith("ggml-"):
        name = name[len("ggml-"):]
    for suffix in QUANT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"mlx-community/whisper-{name}"


def pick_microphone() -> str:
    """Return the first physical avfoundation audio device, skipping virtual ones."""
    listing = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    ).stderr

    in_audio_section = False
    for line in listing.splitlines():
        if "audio devices" in line:
            in_audio_section = True
            continue
        if not in_audio_section or "[" not in line:
            continue
        if any(v in line for v in VIRTUAL_DEVICES):
            continue
        # Lines look like: "[AVFoundation indev @ 0x...] [0] MacBook Pro Microphone"
        match = re.search(r"\]\s*\[\d+\]\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    return ""


def record_until_enter(device: str) -> None:
    """Record a 16kHz mono 16-bit WAV, stopping when the user presses ENTER."""
    # -nostdin so ffmpeg doesn't consume the stdin we read() below.
    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin",
            "-f", "avfoundation", "-i", f":{device}",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(TEMP_AUDIO_FILE), "-y", "-loglevel", "quiet",
        ]
    )

    input()  # block until ENTER
    print("🎤 Stopping recording...")

    # Graceful shutdown (same as Ctrl+C), then wait for ffmpeg to finish writing the file.
    proc.send_signal(signal.SIGINT)
    proc.wait()


def check_local_model(repo: str) -> str:
    """If `repo` is a local path, verify the folder is populated. Returns an error string
    (empty if OK). HF repo IDs like 'org/name' are left for mlx-whisper to download."""
    looks_local = repo.startswith(("/", "./", "../", "~")) or Path(repo).expanduser().exists()
    if not looks_local:
        return ""

    folder = Path(repo).expanduser()
    if not folder.is_dir():
        return f"Model folder not found: {folder}\n   See the README for how to download it."
    if not (folder / "config.json").exists():
        return f"Missing config.json in {folder}"
    if not (folder / "weights.safetensors").exists() and not (folder / "weights.npz").exists():
        return (
            f"Missing weights in {folder}\n"
            f"   Expected weights.safetensors (rename the repo's model.safetensors)."
        )
    return ""


def transcribe(repo: str) -> tuple[str, dict]:
    """Transcribe the recorded WAV. Returns (text, timings) where timings breaks down the
    cost into importing mlx-whisper, loading the model (disk -> GPU, Metal kernel compile),
    and the actual inference — so it's clear where the time goes vs. whisper.cpp."""
    timings = {}

    # Imported here so config/recording errors don't pay the heavy import cost.
    t0 = time.perf_counter()
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder
    timings["import"] = time.perf_counter() - t0

    # Warm ModelHolder with the same dtype transcribe() uses (fp16), so the transcribe()
    # call below reuses this cached model instead of loading it again. This isolates the
    # model-load cost (disk -> GPU, Metal kernel compile) from the inference cost.
    t0 = time.perf_counter()
    ModelHolder.get_model(repo, mx.float16)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(str(TEMP_AUDIO_FILE), path_or_hf_repo=repo)
    timings["inference"] = time.perf_counter() - t0

    return result["text"].strip(), timings


def main() -> int:
    repo = resolve_mlx_repo()
    print(f"Model: {repo}")

    model_error = check_local_model(repo)
    if model_error:
        print(f"❌ Error: {model_error}")
        return 1

    device = pick_microphone()
    if not device:
        print("❌ Error: No physical microphone found.")
        return 1
    print(f"🎙️ Using: {device}")

    print("🔴 Recording... Press ENTER to stop.")
    try:
        record_until_enter(device)

        print("🗣️ Transcribing with local MLX Whisper model...")
        start = time.perf_counter()
        text, timings = transcribe(repo)
        total = time.perf_counter() - start

        if text:
            subprocess.run(["pbcopy"], input=text, text=True)
            print("✅ Transcription copied to clipboard!")
            print("Text:")
            print(text)
        else:
            print("❌ Transcription failed. No text was generated.")
            return 1

        print(
            f"⏱️  import {timings['import']:.2f}s | "
            f"model load {timings['load']:.2f}s | "
            f"inference {timings['inference']:.2f}s | "
            f"total {total:.2f}s"
        )
    finally:
        TEMP_AUDIO_FILE.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
