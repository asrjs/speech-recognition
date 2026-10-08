# C79 — Reviewed browser-code snapshot after C78

Status: G5/G6 real browser integration PASS; Orukeet model acceptance unchanged.

## Reject first, preserve evidence, then verify independently

The C78 source fix `fac133ea` changed only `src/runtime/builtins.ts` in production
source. Its tests/report and `14e13215` discovery docs do not change inference
math. Rechecking the saved C78 browser receipt against the old manifest adapter
produced `FAILED_BROWSER_ORACLE`: `expected_code=false`, all other 18 gates true.
No token, text, PCM, artifact, cleanup or provider mismatch was normalized away.

The same 281 delivered-code paths remain. Eight rows differ: builtins contains
the new dual-failure branch; seven importer rows carry Vite's propagated
`?t=1791430707021` import timestamp. Live local HTTP response bodies for all eight
were reread and matched the C78 receipt's recorded sizes/SHA-256 exactly. These
importers are runtime index/load/local-browser/local-adapter-registry, Parakeet
compat/local-adapter, and the Orukeet smoke JS. Their source files were not edited.
This is a local delivered-byte review, not a universal compiler equivalence proof.

The original `browser-code-contract-jfk.json` is untouched:
`9c736db33165dfcf01accf3c547f1c0bbdeb4efa95232c5a41d9fa8cb3d3d8d9`.
The separate `browser-code-contract-jfk-c78.json` retains the full path set and
records predecessor, reviewed change commit and seed receipt. Its digest is
`857c289a625d53f61047ad3ea618b7c09532aa2b3812f3a53eba5241fe522e35`.
The manifest now explicitly binds that new file. No verifier code or acceptance
threshold changed. Refresh was performed manually after review, not on failure
by an automatic regeneration path. The review was local; no separate subagent
review was available because the collaboration tool reported its thread limit.

## Fresh independent Chrome evidence

Seed capture is C78's direct browser receipt:
`auto-browser-load-failure-fix-2026-10-08-01.json`, SHA
`578880064628a4f0f48b576142108c7543711a8aa5e40aeed1d435bbc83de370`.
It is not the subsequent acceptance run.

Fresh parent: `tools/data/results/orukeet/manifest-chrome-c78-review-2026-10-08-01.json`
SHA `d82bf4dd4567f81fa28e4a6dc5d09ee82f6d8599c2cda030f7ede7d4d1aebdc4`.
Fresh child SHA:
`22309d6bc44d2c5be63c6b15d3eb184321bc1503d46bf76cd3001e95956350dd`.
Exit 0, parent PASS, `PASS_BROWSER_ORACLE_MATCH`, all 19 checks true including
`expected_code`. Chrome 154.0.8037.98, fp32 WebGPU encoder + WASM decoder/joint,
two exact 38-token/text matches against unchanged original oracle and effective
PCM. Disposal and browser closure pass; existing loopback server is untouched.
Recorded first/repeat total times: 3,583.75 / 286.345 ms. These are receipt
observations, not a matched optimization benchmark or a general performance claim.

```powershell
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-c78-new-run.json
python -m unittest discover -s tests -p test_port_manifest.py
python -m unittest discover -s tests -p test_orukeet_browser_manifest.py
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
```

48 lightweight tests PASS. JSON parsing, exact unchanged path-set/eight-row delta
checks and targeted diff checks PASS. Only snapshot data, manifest binding and
docs changed, so the already-passing C78 full runtime suite was not repeated.
Use the reviewed demo on 8767; always choose fresh receipt filenames.

## Limits and next action

Future source/dependency/HMR-server changes can invalidate this local snapshot;
that must fail closed and require another explicit review. Keeping timestamps
in the compared bytes is deliberate; none were stripped to make acceptance pass.
No installed-package, long-audio, streaming, fp16, numerical, quality or soak
acceptance is inferred from this JFK smoke. No weights, engine dependencies or
sibling files were changed/downloaded. Return to Orukeet's earliest unresolved
feature arithmetic gate using the existing boundary probes; do not rerun the
rejected C74 output-component cast. Redux exact-runtime oracle remains a separate
dependency, not a reason to substitute teacher weights or declare it ported.
