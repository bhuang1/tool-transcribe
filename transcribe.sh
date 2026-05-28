#!/bin/bash

# --- Configuration ---
# Set the path to your whisper.cpp directory and model file.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# --- Script ---
# High-resolution clock (macOS bash 3.2 has no EPOCHREALTIME; perl is always present).
now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }
SCRIPT_START=$(now)

MODEL_PATH="$WHISPER_CPP_DIR/models/$WHISPER_MODEL.bin"
echo "Model path: $MODEL_PATH"
MAIN_EXECUTABLE="$WHISPER_CPP_DIR/build/bin/whisper-cli"
TEMP_AUDIO_FILE="temp_audio.wav"

# Check if the whisper-cli executable exists
if [ ! -f "$MAIN_EXECUTABLE" ]; then
  echo "❌ Error: whisper-cli not found at $MAIN_EXECUTABLE"
  echo "Please build whisper.cpp by running 'cmake -B build' and 'cmake --build build' in its directory."
  exit 1
fi

echo "🔴 Recording... Press ENTER to stop."

# 1. Pick the best physical microphone (skip virtual devices like Zoom/Teams).
AUDIO_DEVICE=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 |
  grep "audio devices" -A 99 |
  grep -v "ZoomAudioDevice\|Microsoft Teams\|BlackHole\|Soundflower\|virtual\|AVFoundation audio devices" |
  grep "\[" |
  sed 's/.*\] //' |
  head -1)

if [ -z "$AUDIO_DEVICE" ]; then
  echo "❌ Error: No physical microphone found."
  exit 1
fi

echo "🎙️ Using: $AUDIO_DEVICE"

# 2. Record a 16-bit WAV file, as required by whisper-cli.
#    Use -nostdin so ffmpeg doesn't consume stdin (which conflicts with read).
ffmpeg -nostdin -f avfoundation -i ":$AUDIO_DEVICE" -ar 16000 -ac 1 -c:a pcm_s16le "$TEMP_AUDIO_FILE" -y -loglevel quiet &
FFMPEG_PID=$!

# Wait for the user to press Enter
read -r

echo "🎤 Stopping recording..."

# 2. Send a graceful shutdown signal (SIGINT, same as Ctrl+C) to ffmpeg.
kill -INT "$FFMPEG_PID"

# 3. CRITICAL STEP: Wait for the ffmpeg process to completely finish.
#    The script will pause here until ffmpeg is done writing the file.
wait "$FFMPEG_PID"

echo "🗣️ Transcribing with local Whisper model..."

# 4. Transcribe the audio file and swallow all system logs.
#    whisper-cli loads the model and runs inference in one shot, so this measures both.
TRANSCRIBE_START=$(now)
TRANSCRIPTION=$("$MAIN_EXECUTABLE" -m "$MODEL_PATH" -f "$TEMP_AUDIO_FILE" -nt 2>/dev/null)
TRANSCRIBE_TIME=$(perl -e "printf '%.2f', $(now) - $TRANSCRIBE_START")

# Optional: Strip the leading space whisper.cpp sometimes adds
TRANSCRIPTION=$(echo "$TRANSCRIPTION" | xargs)

# 5. Check if transcription was successful before copying.
if [[ -n "$TRANSCRIPTION" ]]; then
  echo "$TRANSCRIPTION" | pbcopy
  echo "✅ Transcription copied to clipboard!"
  echo "Text:"
  echo "$TRANSCRIPTION"
else
  echo "❌ Transcription failed. No text was generated."
fi

TOTAL_TIME=$(perl -e "printf '%.2f', $(now) - $SCRIPT_START")
echo "⏱️  transcribe (load+inference) ${TRANSCRIBE_TIME}s | total ${TOTAL_TIME}s"

# 6. Clean up the temporary audio file.
rm "$TEMP_AUDIO_FILE"
