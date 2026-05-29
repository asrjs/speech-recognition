#!/bin/bash
# Whisper splitgraph WebGPU smoke server
# Serves model files, fixtures, pre-computed mel, and smoke page on localhost:8765
#
# Usage:
#   npm run build  # first, if mel generation needed
#   ./scripts/whisper-webgpu-smoke.sh
#   Then open http://localhost:8765/tests/smoke/whisper-webgpu-smoke.html

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${WHISPER_WEBGPU_MODEL:-/tmp/whisper-base-4graph/fp16}"
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

# Generate mel reference for jfk2.en.wav (uses library WhisperMelProcessor, not hand-rolled DFT)
MEL_REF="tests/smoke/jfk2-mel.json"
if [ ! -f "$MEL_REF" ] || [ "tests/fixtures/jfk2.en.wav" -nt "$MEL_REF" ]; then
  echo "Generating mel reference for jfk2.en.wav..."
  node --input-type=module -e "
import { WhisperMelProcessor } from './dist/audio/whisper-mel.js';
import { readFileSync, writeFileSync } from 'node:fs';
function readWavMono(p) {
  const b = readFileSync(p);
  let off=12,fmt=null,data=null;
  while(off+8<=b.length){
    const id=b.toString('ascii',off,off+4), sz=b.readUInt32LE(off+4), st=off+8;
    if(id==='fmt ') fmt={channels:b.readUInt16LE(st+2),sampleRate:b.readUInt32LE(st+4),bitsPerSample:b.readUInt16LE(st+14)};
    else if(id==='data') data=b.subarray(st,st+sz);
    off=st+sz+(sz%2);
  }
  if(!fmt||!data) throw new Error('bad wav');
  const bytes=fmt.bitsPerSample/8, frames=Math.floor(data.length/bytes/fmt.channels);
  const out=new Float32Array(frames);
  for(let i=0;i<frames;i++){let s=0;for(let ch=0;ch<fmt.channels;ch++){const p=(i*fmt.channels+ch)*bytes;s+=data.readInt16LE(p)/32768}out[i]=s/fmt.channels}
  return fmt.sampleRate===16000?out:(()=>{const r=16000/fmt.sampleRate,o=new Float32Array(Math.max(1,Math.floor(out.length*r)));for(let i=0;i<o.length;i++){const x=i/r,x0=Math.floor(x),x1=Math.min(out.length-1,x0+1),t=x-x0;o[i]=(out[x0]??0)*(1-t)+(out[x1]??0)*t}return o})();
}
const samples = readWavMono('tests/fixtures/jfk2.en.wav');
const mel = new WhisperMelProcessor({ nMels: 80 });
const result = mel.process(samples);
const padded = WhisperMelProcessor.padToFrames(result, 3000);
writeFileSync('$MEL_REF', JSON.stringify(Array.from(padded)));
console.log('Mel reference written: ' + padded.length/80 + ' frames');
" 2>/dev/null || echo "WARN: mel generation failed (run 'npm run build' first)"
  echo ""
fi

echo ""
echo "Open: http://localhost:$PORT/tests/smoke/whisper-webgpu-smoke.html"
echo "Press Ctrl+C to stop."
echo ""

python3 -m http.server "$PORT" --bind 127.0.0.1
