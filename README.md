# tool-transcribe

Record speech from your microphone, transcribe it locally, and copy the text to your
clipboard. Press ENTER to stop recording — nothing leaves your machine.

Two interchangeable engines are provided:

| Script          | Engine                   | Acceleration            |
| --------------- | ------------------------ | ----------------------- |
| `transcribe.sh` | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Metal GPU / ANE (needs a local C++ build) |
| `transcribe.py` | [mlx-whisper](https://pypi.org/project/mlx-whisper/) | Metal GPU via Apple [MLX](https://github.com/ml-explore/mlx) (no build step) |

A third script, **`parse-and-transcribe.sh`**, chains whisper.cpp with a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) LLM to additionally **clean up** the
transcript — stripping "um"/"ah" filler, stutters, and repetition, fixing punctuation, and
reflowing into Markdown paragraphs — without summarizing or rephrasing. See its section below.

All run entirely offline and target macOS on Apple Silicon.

## Prerequisites

- **macOS on Apple Silicon** (M-series) — required by both MLX and whisper.cpp's Metal backend.
- **ffmpeg** — used to record from the microphone:

  ```sh
  brew install ffmpeg
  ```

- **[uv](https://docs.astral.sh/uv/)** — only for `transcribe.py`:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Usage — `transcribe.py` (mlx-whisper)

```sh
uv run transcribe.py
```

That's it. `transcribe.py` is a [PEP 723](https://peps.python.org/pep-0723/) script with its
dependency (`mlx-whisper`) declared inline, so `uv` automatically creates an isolated
environment and installs it on first run — there is no separate `pip install` or virtualenv to
manage. On first run the model is also downloaded automatically from Hugging Face
(`large-v3-turbo` is ~1.6 GB), then cached for subsequent runs.

The flow:

1. The best physical microphone is detected (virtual devices like Zoom/Teams/BlackHole are skipped).
2. Recording starts — speak, then press **ENTER** to stop.
3. The audio is transcribed locally on the GPU.
4. The text is printed and copied to your clipboard.

### Configuration

The model is configured in `env.sh` (shared with `transcribe.sh`):

- **`MLX_WHISPER_MODEL`** — if set, used directly. It can be an `mlx-community` Hugging Face
  repo **or a local model folder**. The default is `mlx-community/whisper-large-v3-turbo-8bit`
  (the MLX analog of the whisper.cpp `q8_0` model).
- **`WHISPER_MODEL`** — used only when `MLX_WHISPER_MODEL` is unset. The whisper.cpp name is
  mapped to an `mlx-community` repo by stripping the `ggml-` prefix and any quantization suffix:

  ```
  ggml-large-v3-turbo-q8_0  ->  mlx-community/whisper-large-v3-turbo
  ggml-base-q5_1            ->  mlx-community/whisper-base
  ```

An inline env var still wins over `env.sh` for one-off runs:

```sh
MLX_WHISPER_MODEL=mlx-community/whisper-large-v3-turbo-4bit uv run transcribe.py
```

### Using a pre-downloaded model (offline — this is the default)

`env.sh` defaults `MLX_WHISPER_MODEL` to a **local folder** so nothing is downloaded at
runtime. Populate that folder once (`models/` is gitignored):

```sh
cd models/whisper-large-v3-turbo-8bit   # mkdir -p it first

# Download only the two files mlx-whisper needs:
hf download mlx-community/whisper-large-v3-turbo-8bit config.json       --local-dir .
hf download mlx-community/whisper-large-v3-turbo-8bit model.safetensors --local-dir .

# mlx-whisper 0.4.3 looks for the weights as `weights.safetensors`, so rename:
mv model.safetensors weights.safetensors
```

The folder must end up containing:

| File | Notes |
| --- | --- |
| `config.json` | model dimensions + quantization config |
| `weights.safetensors` | the weights — renamed from the repo's `model.safetensors` |

You do **not** need `multilingual.tiktoken` (mlx-whisper ships its own tokenizer vocab) or the
repo's `README.md` / `.gitattributes`.

> **Why the rename?** mlx-whisper `0.4.3` (the version `uv` installs) only looks for
> `weights.safetensors` / `weights.npz`. Newer source also accepts `model.safetensors`
> directly, but the released version does not — so rename to be safe.

`MLX_WHISPER_MODEL` can also be a HF repo (e.g. `mlx-community/whisper-large-v3-turbo-8bit`)
if the machine *can* reach Hugging Face. Available `large-v3-turbo` variants include `-8bit`
(≈ q8), `-4bit`, `-fp16`, and the default `mlx-community/whisper-large-v3-turbo`.

## Usage — `transcribe.sh` (whisper.cpp)

Requires a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp) checkout, built with
`cmake -B build && cmake --build build`, plus the model `.bin` downloaded into its `models/`
directory. Point `env.sh` at it via `WHISPER_CPP_DIR` and `WHISPER_MODEL`, then:

```sh
./transcribe.sh
```

## Usage — `parse-and-transcribe.sh` (whisper.cpp → llama.cpp cleanup)

Same first pass as `transcribe.sh`, then a **second pass** that runs the raw transcript
through a local LLM to remove verbal filler, fix punctuation, and reflow into Markdown
paragraphs — **without** summarizing, paraphrasing, or rewording. The cleaned text is copied
to the clipboard and saved to a `.md` file.

```sh
./parse-and-transcribe.sh                   — mic recording, no .md file saved
./parse-and-transcribe.sh recording.m4a     — transcribe file, no .md file saved
./parse-and-transcribe.sh -o                — mic recording, .md file saved
./parse-and-transcribe.sh -o recording.m4a  — transcribe file, .md file saved
```

With a file argument the `.md` is saved next to it (`meeting.m4a` → `meeting.md`); for mic
input it's saved as `transcript-<timestamp>.md` in the script directory.

### Prerequisites

In addition to the whisper.cpp setup above, you need **llama.cpp**:

```sh
brew install llama.cpp
```

> **Use the `llama-completion` binary, not `llama-cli`.** Recent llama.cpp builds reject the
> non-interactive `-no-cnv` flag in `llama-cli` ("use llama-completion instead") and drop into
> a chat REPL. `env.sh` therefore points `LLAMA_CLI` at `llama-completion`.

### Configuration (`env.sh`)

- **`LLAMA_CLI`** — path to the `llama-completion` binary (default
  `/opt/homebrew/bin/llama-completion` from Homebrew; a local build is at
  `<llama.cpp>/build/bin/llama-completion`).
- **`LLAMA_MODEL_PATH`** — path to the cleanup LLM (a `.gguf` file). Default is
  `gemma-3-12b-it-Q4_K_M.gguf`.

**Model choice matters.** Use a **non-thinking** instruction-tuned model. Thinking models
(e.g. the gemma-4 reasoning MoE) blow the latency budget with mandatory chain-of-thought, and
forcing thinking off makes them loop or hallucinate. Good non-thinking options:

| Model | Notes |
| --- | --- |
| `gemma-3-12b-it` (Q4_K_M) | The default. Non-thinking, follows the cleanup rules well. ~5–6 s total on an M2 Max. |
| `Qwen3-4B-Instruct-2507` (Q6_K) | Faster (smaller, ~sub-1 s load). True non-thinking instruct model. `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`. |
| `Meta-Llama-3.1-8B-Instruct` (Q6_K) | Very loop-resistant, strong instruction-following. `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`. |

The cleanup prompt and the (tunable) llama.cpp sampler flags — low temperature, light
repetition/presence penalties, a token cap, and a `-r "<<<"` stop sequence that prevents the
model fabricating extra text — are documented inline near the top of the `# --- Step 3` block
in the script.

### Run from anywhere (`.zshrc` alias)

The script resolves its own directory, so it sources `env.sh` and finds the binaries no matter
where it's called from. Add an alias to `~/.zshrc`:

```sh
alias cleantranscribe="/Users/bettyhuang/myapps/tool-transcribe/parse-and-transcribe.sh"
```

Then `cleantranscribe` (mic) or `cleantranscribe path/to/audio.m4a` (file) works in any directory.
