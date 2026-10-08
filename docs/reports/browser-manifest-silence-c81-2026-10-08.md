# C82 — Reviewed current-code Chrome manifest and silence case

Status: bounded G1/G5/G6 browser output smoke integration PASS for speech/silence.
Full numerical/model/first-delivery acceptance remains incomplete.

## Change and evidence boundary

The current manifest now includes `pure-silence-1s` and
`chrome-oracle-pure-silence-webgpu-hybrid`, binding the existing one-second
WAV, original NeMo oracle and effective float32 PCM by their actual SHA-256.
The fixture expects explicit empty tokens/text, not missing inference records.
No original audio, oracle, weights or graph was edited. Corrected a copied
oracle digest typo in the C81 report; the file and receipts were already correct.

Before rebinding, comparison of the final C81 silence receipt against the
matching fixture and C78 contract rejected only `expected_code`; all other
18 checks passed. The first exploratory comparison against the JFK fixture
also correctly rejected audio/oracle/PCM/output mismatches; that is not evidence
of a model regression. A subsequent hand-copied digest typo was checked against
the live file and corrected before adding the fixture.

C78 contract stays unchanged at SHA
`857c289a625d53f61047ad3ea618b7c09532aa2b3812f3a53eba5241fe522e35`.
Separate `browser-code-contract-c81.json` SHA:
`3505620091481a3e6695b207968e6d7b5c69a8cf9396349b3a242bf735c71ede`.
It names predecessor, reviewed commits `b6d6794a` / `18a5e43d`, and final
C81 seed `auto-browser-pure-silence-hybrid-2026-10-08-03.json` SHA
`af5263b768a0e08075afea81bde86f6f9f5f033f8a030ddb26055bf40171319e`.

Same 281 unique paths, 288 deliveries with consistent duplicate bodies, exactly
eight changed rows: builtins and smoke JS from the two source commits, plus
six unchanged-source importers carrying propagated Vite timestamps
`?t=1791432577341`. All eight live response sizes/hashes matched the seed.
The importers are runtime index/load/local-browser/local-adapter-registry and
Parakeet compat/local-adapter. Timestamp-bearing bytes were not normalized.
Independent read-only review approved the snapshot/manifest/test scope.
Neither runner, adapter, numerical tolerance nor any of the 19 gates changed.

## Validation

New adapter simulation checks valid explicit silence and eight rejecting
mutations: no inference records, absent/null/wrong token IDs, hallucinated text,
missing/changed served code and incomplete disposal. It writes only a temporary
synthetic oracle; this is not original-model or GPU evidence.
Adapter tests 6 PASS, manifest tests 24 PASS, browser runner tests 19 PASS,
demo tests 13 PASS. Python syntax and scoped diff checks pass. No production
runtime changed, so the C81 full-suite result is dated context, not a new run.

Fresh independent manifest-driven silence receipt:
`tools/data/results/orukeet/manifest-chrome-c81-silence-2026-10-08-01.json`, SHA
`92ab11ab226803cb92f118fc1fb4cd05e5107b132336261051c251dd700470c0`.
Child SHA `2758102f7733b7e303b9ec4a2a1fdad6cec517dd6a47f8ba61f370636c3e2623`.
Exit 0, parent PASS, `PASS_BROWSER_ORACLE_MATCH`, all 19 checks true.
Two actual empty token/text outputs match the original effective PCM; Chrome
154.0.8037.98, fp32 WebGPU encoder + WASM decoder/joint and JS processor.
Awaited disposal/browser closure pass. Cold/repeat totals 3265.985 / 95.48 ms
are observations, not a matched speed claim. The existing server is untouched.

Fresh JFK control on the same final code:
`tools/data/results/orukeet/manifest-chrome-c81-jfk-2026-10-08-01.json`, SHA
`195955e525bab9e91be8fe833725c45078d5262b21df0fa4483483592bee4bd1`.
Child SHA `baf82bd010c535ae2a2e6696f8f5e05afeb96b245be4cad36e70b5abf7f17e0a`.
Exit 0, parent PASS, all 19 checks true; two exact 38-token/text outputs with
original JFK PCM, same actual hybrid placement and awaited cleanup. Totals
2144.655 / 259.96 ms are dated observations only. This is separate from the
silence seed and revalidates speech after the final C81 controller refinement.

## Reproduction and remaining scope

From the repository root with the reviewed demo server on 8767:

```powershell
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-pure-silence-webgpu-hybrid --output tools/data/results/orukeet/manifest-silence-new.json
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-jfk-new.json
```

Choose fresh output names, never overwrite receipts. These cases launch Chrome,
load existing local graphs, call the library, compare complete outputs and save
parent/child JSON; no download or second decoder. TLS remains enabled except
the established explicit loopback dev-certificate policy.

This local delivered-byte contract is not universal compiler equivalence or
installed-package acceptance. Strict production feature numerics still fail;
output matches do not erase normalization differences. Streaming, fp16,
held-out quality, silence soak and GPU memory recovery remain separate gates.
Future code/server/dependency changes must fail the snapshot until explicitly
reviewed. Return to the next distinct product/model gate, not automatic snapshot
refresh or the already-rejected FFT hypotheses. Qwen's isolated coordination
file still reports numerical FAIL and browser BLOCKED/UNTESTED; it is not promoted.
