# Non-Turbo Whisper Large v3 scope stop

On 2026-10-25 the user explicitly removed `openai/whisper-large-v3`
(non-Turbo) from active work and redirected effort to newer ASR models suited
to browser WebGPU/WASM. Existing Whisper Large v3 Turbo custom graph work is
separate and remains preserved.

The parent stopped the verified metrics test Python PID 7144 and its descendants;
no unrelated browser or preview server was stopped. Process inventory:
`../../../demos/tmp/real-e2e/whisper-large-v3-resume2-stopped-processes.json`.

No new Large inference, export, graph optimization, or memory repair is queued.
Existing models, source edits, exports and failed receipts remain intact.
The interrupted metrics run is UNVERIFIED, not a model acceptance result:
`../../../demos/tmp/real-e2e/remaining-models-20261024/whisper-large-v3-v16-pcm-metrics-resume2/realtime/`.

The prior restart failed because the demo preview server was offline; that
separate failure remains in `whisper-large-v3-v16-pcm-metrics-resume/`.

A diagnostic-tool issue was identified during this run: the PCM binding writes
raw bytes, then resolves the module-global `write_receipt`, which wrappers
replace with synchronous browser/worker evaluation. This can stall the binding
before its metadata JSON is persisted. The module globals identity was checked;
this is not evidence of a model inference failure. No acceptance claim is made
from that interrupted capture. Tool repair can be resumed when needed for an
in-scope model, independently of the excluded Large model.

Active direction remains newer ASR model loading, real-weight file/capture/VAD
and native-streaming validation across the four demos. Existing unverified
quality and numerical gates remain unchanged. Physical microphone and remote
CI have not been certified by this checkpoint. New bot PR #238 was discovered
and is under review; it has not been merged.
