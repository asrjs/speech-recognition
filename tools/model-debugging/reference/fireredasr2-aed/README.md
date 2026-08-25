# FireRedASR2-AED reference tooling

This folder captures the upstream FireRedASR2-AED implementation as a local
reference. It does not download checkpoints and it does not add a runtime
preset by itself.

## Capture a native reference

The source checkout and the four upstream AED files must already exist:

~~~powershell
$PYTHON = 'C:\path\to\firered\python.exe'
$FIRERED = 'N:\github\ysdede\FireRedASR2S'
$MODEL = 'N:\models\FireRedASR2-AED'

& $PYTHON tools/model-debugging/reference/fireredasr2-aed/capture_firered_reference.py ~
  --fire-red-root $FIRERED ~
  --model-dir $MODEL ~
  --audio tools/data/fixtures/audio/jfk-short.wav ~
  --output tools/data/results/firered/fireredasr2-aed-reference.json ~
  --device cpu ~
  --batch-size 1 ~
  --beam-size 3 ~
  --timestamps
~~~

The JSON records:

- the checkpoint file SHA-256 values and FireRed source revision;
- the Kaldi fbank/CMVN contract and stable sample identity;
- token IDs, token pieces, native text, confidence, batch order, and latency;
- optional CTC-refined token timestamps.

The script intentionally fails if feature extraction drops or reorders an
input. That keeps mixed-length batch comparisons from silently comparing the
wrong rows.

## Export and verify ONNX boundaries

The checked-in exporter consumes the same local checkpoint and writes separate
encoder, full-prefix AED decoder, and CTC graphs:

~~~powershell
& $PYTHON tools/model-debugging/reference/fireredasr2-aed/export_firered_onnx.py ~
  --fire-red-root $FIRERED ~
  --model-dir $MODEL ~
  --output-dir N:\models\onnx\fireredasr2-aed ~
  --dtype float32
~~~

Capture a reference with the same fixture and compare the exported stages:

~~~powershell
& $PYTHON tools/model-debugging/reference/fireredasr2-aed/verify_firered_onnx.py ~
  --model-dir N:\models\onnx\fireredasr2-aed ~
  --reference tools/data/results/firered/fireredasr2-aed-reference.json ~
  --output tools/data/results/firered/fireredasr2-aed-onnx-parity.json
~~~

The verifier checks encoder states, lengths, masks, teacher-forced decoder
logits, and CTC log probabilities when timestamps were captured. A beam
reference is retained as the quality oracle; the verifier reports greedy
token equality separately because greedy and beam output are not expected to
match for every sample.

## Audit an ONNX bundle

After an export, audit every graph and its co-located files from Node:

~~~powershell
node tools/model-debugging/scripts/node-audit-onnx-artifact.mjs ~
  --model-dir N:\models\onnx\fireredasr2-aed ~
  --output tools/data/results/firered/fireredasr2-aed-onnx-audit.json
~~~

The audit hashes all files, creates each ONNX graph with the native CPU
execution provider, records graph input/output names, and reports likely
external-data sidecars. It is strict by default and never fills missing
files. Add --allow-load-failures only when collecting a diagnostic report.

## Export boundary

The upstream checkout contains an encoder TensorRT export helper, but the
browser port still needs independent encoder, AED decoder, and CTC graph
contracts. Do not treat an encoder-only TensorRT export as a complete
FireRed runtime. The next parity step is to compare the captured native JSON
with each graph on the same batch-1 and mixed-length fixtures before adding
src/models/firered-aed.
