#!/bin/bash

# --- Configuration ---
# Set the path to your whisper.cpp directory and model file.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# --- Script ---
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

# 1. Record a 16-bit WAV file, as required by whisper-cli.
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -c:a pcm_s16le "$TEMP_AUDIO_FILE" -y -loglevel quiet &
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
TRANSCRIPTION=$("$MAIN_EXECUTABLE" -m "$MODEL_PATH" -f "$TEMP_AUDIO_FILE" -nt 2>/dev/null)

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

# 6. Clean up the temporary audio file.
rm "$TEMP_AUDIO_FILE"
