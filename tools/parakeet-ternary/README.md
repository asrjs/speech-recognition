# Packed ternary diagnostic tools

Engineering-only tools, not a supported ASR executor. Do not use the historical
teacher-correlation/substitution route as a Redux model port: its failed receipt
is preserved. A small file is not a smaller parameter model or a measured runtime win.

## Offline format preflight

`audit_packed.py` requires recorded archive, manifest and config SHA-256 values.
It hashes and consumes the same immutable byte snapshots (archive tensors are
views into the snapshot), so a later path replacement cannot change audited bytes.
This holds the compressed archive in host RAM; it does **not** expand 604M weights.
It checks the manifest/config module set, group/packing/weight-rule contract,
qweight/scales shape and dtype, code range, zero padding, finite scales and count.
Only `thrush-ternary-v2` / HF names / fp16 scales are supported; unknown formats fail.

Verified local Redux invocation on 2026-10-08:

```powershell
python tools/parakeet-ternary/audit_packed.py --archive .cache/parakeet-redux/model.safetensors --archive-sha256 78ec25733ee0d0c1586d1346fc86db9d0c2e436e3a8ab1d32a82d1bb8f848d21 --manifest .cache/parakeet-candidates/parakeet-redux-ternary.json --manifest-sha256 1221c6d3ce901ffe09c089da758a8db8b76189f80cff41c5afc244fc61e2051d --config .cache/parakeet-candidates/parakeet-redux-config.json --config-sha256 503c653b2e3bb788adbcb04f5abdee532d958686564081baeed133ff10143f6e
python -m unittest discover -s tools/parakeet-ternary -p test_audit_packed.py
```

The new synthetic tests need no model/download/cache. The older `test_ternary.py`
also contains a cached-manifest integration test, so the full folder suite needs
that optional cache. Local full-folder result: 28 tests PASS. Real preflight:
264 modules, 120,913,920 qweight bytes, 9,437,184 scale bytes. Its
`PASS_PACKED_FORMAT_ONLY` does not certify official inference, decoder correctness,
repacking, activation quantization, numerical parity, browser memory or speed.

## Source/runtime boundary

Local archive/config/manifest match the live HF snapshot
[`2bf128600aac4b16946f7ed8372e56117fe5e23b`](https://huggingface.co/moondream/parakeet-redux/tree/2bf128600aac4b16946f7ed8372e56117fe5e23b).
The previously recorded `efec50629e2e690eb1dc58115d46a3dcdfd20442` URL returns 404;
do not silently bind a receipt to that old address. Immutable file hashes remain
the content identity even if upstream history changes.

Official Kestrel
[`weights.py` at `8e6fb899bb8a076f3235b6a5ab832a78c031c77b`](https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/kestrel/models/parakeet_tdt/weights.py)
converts five-trit base-3 file rows into four-code two-bit kernel rows, fuses
q/k/v rows and relative-position projections, loads the complete state strictly,
and attaches VAD when present. CPU/Metal retain packed representations; CUDA
materializes dense weights. The estimated **code buffers alone** after two-bit
repacking are 150,994,944 bytes, not total resident memory.

The exact packing/compute implementation is delegated to `kestrel_kernels`;
do not infer bit-order or activation arithmetic from public loader geometry.
The loader's outer activation dtype statement and the
[ASR docs' CPU int8 activation description](https://github.com/m87-labs/kestrel/blob/8e6fb899bb8a076f3235b6a5ab832a78c031c77b/docs/asr.md)
need a kernel/reference run to reconcile. No kernel equivalence is established.

Next: original recommended-runtime oracle, pinned effective PCM and stage
captures before faithful graph export or a packed WebGPU experiment. The full
pinned-commit contract map (features/subsampler/relpos fusions/decoder/joint/
VAD/segmentation) is recorded in
`../data/results/model-candidates/redux-kestrel-contract-2026-10-08.json`.
Local Transformers5 environment has no `moondream`, `kestrel` or `kestrel_kernels` module;
that check applies only to this environment, not every workstation environment.
No install or large download was performed. Coordinate heavy work with Qwen's
separate CPU/RAM window. Receipt:
`tools/data/results/model-candidates/redux-source-bound-preflight-2026-10-08.json`.
