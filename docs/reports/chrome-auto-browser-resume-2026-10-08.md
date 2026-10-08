# Automatic Chrome model test: current-source recheck

User-requested recovery of URL-driven model loading, inference and automatic
result saving. The existing `tests/smoke/orukeet-auto-browser.py` is operational;
no new inference loop or production workaround was needed. Added its discovery
route and reproducible command to `tools/browser-validation/README.md`.

Fresh receipt:
`tools/data/results/orukeet/auto-browser-user-resume-2026-10-08-01.json`

Receipt SHA-256:
`88d0013fb7643b4d73a657ad14db1a8a01fb2923f579211ee0f0667181d7daa4`

Result: `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`.
Installed Chrome 154.0.8037.98 used the already-running, HTTP-200-verified
loopback demo at `https://127.0.0.1:8767/orukeet-smoke.html`. An attempted fresh
server start failed safely on the occupied port; the existing server was not
stopped or modified. The runner appended `auto=1` and injected local files.

- Own packed Orukeet fp32 artifacts; WebGPU encoder, WASM decoder/joint.
- JFK fixture, two independent transcriptions, 38 exact original-oracle tokens
  and full text each time.
- Official effective float32 PCM digest matched.
- Delivered-code binding, execution, controller and timing gates passed.
- Awaited model disposal and own Chrome closure passed.
- Eighteen runner regression tests passed; targeted documentation diff checked.

The process emitted an inherited `NODE_TLS_REJECT_UNAUTHORIZED=0` warning.
No global environment or certificate configuration was changed. The runner
restricts browser traffic to its explicit loopback origin; this test does not
certify external TLS behavior.

Scope: real source-checkout browser smoke and same-input token/text evidence,
not complete feature/encoder numerical parity, fp16 acceptance, native streaming,
held-out quality or memory soak. The Orukeet adapter is model-specific. Historical
`webgpu-agent-test` and the manifest browser warm-up path are separate surfaces;
warm-up-only receipts must not be represented as inference/oracle acceptance.
