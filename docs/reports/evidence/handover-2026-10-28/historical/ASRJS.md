# ASRJS execution profile

`asrjs-manifest.json` is exported from the library's approved profile registry.
It pins existing graph, config and vocabulary files at immutable artifact
revision f88260fa0777fe0868dda6df85d1a98f012a4a7a. The manifest commit is a
metadata publication; it does not change or republish the model weights.

Validated profile: FP16 encoder on WebGPU; INT8 decoder/joint on WASM;
JavaScript nemo128 audio processor. Receipt scope: Parakeet TDT v3 on the English
JFK fixture, offline transcription and controlled prerecorded Chrome microphone
with manual/TEN-VAD capture and disposal. Physical microphone hardware, every
language, every audio duration and numerical parity at each stage remain
unverified. Other graph/precision/backend combinations in this repository are
not approved by this profile merely because their files exist.

Evidence: https://github.com/asrjs/speech-recognition/blob/abf5d6dd90f04e56a0d102e0fd7439a2d92a8590/docs/reports/evidence/microphone-composition-2026-10-25/real-run-final/receipt.json

License and attribution remain in README.md and ATTRIBUTION.md. Original model:
NVIDIA Parakeet, CC-BY-4.0. ONNX baseline: istupakov/parakeet-tdt-0.6b-v3-onnx
revision 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce. Local FP16 conversion and
ORT cast fixes are described in the existing provenance records. The new
manifest's file hashes identify the current artifacts; older historical
provenance includes a stale config size and is retained as historical evidence.
