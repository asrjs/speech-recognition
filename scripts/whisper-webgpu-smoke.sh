#!/bin/bash
# Whisper splitgraph WebGPU smoke server
# Serves model files, fixtures, and smoke page on localhost:8765
#
# Usage:
#   ./scripts/whisper-webgpu-smoke.sh
#   Then open http://localhost:8765/tests/smoke/whisper-webgpu-smoke.html

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${WHISPER_WEBGPU_MODEL:-/tmp/whisper-base-4graph/fp16}"
FIXTURE_DIR="tests/fixtures"
PORT="${WHISPER_WEBGPU_PORT:-8765}"

echo "=== Whisper WebGPU Smoke Server ==="
echo "Model:  $MODEL_DIR"
echo "Port:   $PORT"
echo ""

# Verify model exists
for f in encoder_model.onnx decoder_init.onnx decoder_step.onnx manifest.json tokenizer.json; do
  if [ ! -f "$MODEL_DIR/$f" ]; then
    echo "ERROR: missing $MODEL_DIR/$f"
    exit 1
  fi
done

# Create symlink to model under tests/smoke/ so HTTP server can reach it
MODEL_LINK="tests/smoke/.whisper-fp16-model"
rm -f "$MODEL_LINK"
ln -s "$(realpath "$MODEL_DIR")" "$MODEL_LINK"

echo "Model symlink: $MODEL_LINK -> $MODEL_DIR"
echo ""
echo "Open: http://localhost:$PORT/tests/smoke/whisper-webgpu-smoke.html"
echo "Press Ctrl+C to stop."
echo ""

python3 -m http.server "$PORT" --bind 127.0.0.1
