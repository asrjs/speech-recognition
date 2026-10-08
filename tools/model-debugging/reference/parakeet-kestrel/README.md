# Pinned Kestrel reference environment (Redux/Ultra oracle)

User-approved dedicated venv (2026-10-08); it does not touch the NeMo or Qwen
environments. Stack: `kestrel @ git+https://github.com/m87-labs/kestrel@8e6fb899bb8a076f3235b6a5ab832a78c031c77b`,
`kestrel-kernels==0.7.5`, `kestrel-native==0.1.9`, CPU torch 2.14.1 on Windows
CPython 3.11. Local snapshots (hashes in the receipts):

- Redux `.cache/parakeet-redux-full` at `moondream/parakeet-redux@2bf128600aac4b16946f7ed8372e56117fe5e23b`
  (archive `78ec2573…` matches the long-pinned preflight identity).
- Ultra `.cache/parakeet-ultra` at `moondream/parakeet-ultra@73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b`.
- int4 `.cache/parakeet-int4` at `efederici/parakeet-tdt-0.6b-v3-onnx-int4@9828da0021d81df726b3d5ddb2d6d04354942749`.

## Workstation limit: no packed CPU kernel

This Xeon W-2135 (Skylake-W) has AVX2/AVX512 but no VNNI. The kernel bundle's
auto probe reports `'scalar'`; `set_gemm_isa('avx2')` is accepted but the gemm
raises `no 'avx2' matrix-multiply path on this machine`. Packed `gemm8` compute
therefore cannot run here at all. Both scripts override `resident_form` to the
official **dense/oracle** branch (the form CUDA gets; `TernaryWeight.dequantized`
is the single place codes become floats). Output oracles from this machine are
dense-oracle evidence; packed-kernel speed or numerics need a VNNI CPU or CUDA.

## Scripts

- `kernel_parity_probe.py` — official dense-oracle dequantization vs this repo's
  independent `tools/parakeet-ternary/ternary.py` reader. Result 2026-10-08:
  `PASS_PACK_PARITY_BIT_EXACT` on 4 modules (weights and forwards bit-exact;
  fused qkv/relative projections excluded by design). Receipt:
  `tools/data/results/model-candidates/redux-kernel-parity-2026-10-08.json`.
- `redux_oracle.py` — exact-weight official transcribe of the pinned JFK
  effective PCM (`ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`),
  dense-oracle form, output-only receipt.

These are original-runtime evidence for the model lane, not ports; no export,
WebGPU claim or library promotion follows from them. Keep heavy runs single
(coordinate with the Qwen agent's CPU/RAM window).
