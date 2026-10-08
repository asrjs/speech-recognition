# C72 — Compact browser consumers from the npm archive

Status: G3/G6 bounded package-consumer PASS; installation/inference separate.

The existing source compactness probe now also accepts `consumerRoot`, resolving
`@asrjs/speech-recognition` through the isolated package's public root export.
Default source mode is unchanged. Missing/wrong package identity fails; the same
32,768-byte budget, no external imports and no retained model/preset/ORT/native
rules apply to both source and compiled `dist` paths. Visited-but-tree-shaken
code remains distinct from retained code. Metadata, tool and loaded JS/TS bytes
are hash-bound and rechecked after measurement.

`packed-native-absent-smoke.mjs` reuses its existing offline npm pack/extraction
and enumerated local dependency links, measures both consumers, and writes an
exclusive new compact receipt in the retained fixture before continuing its
existing native-absence and strict declaration checks. No second pack or install.

## Real measured result

Fresh current-tree build and offline archive consumption PASS:

| Root export | Minified browser bytes | Budget |
| --- | ---: | ---: |
| createSpeechRuntime | 21,433 | 32,768 |
| transcribeWithWindowing | 19,772 | 32,768 |

Both outputs have zero imports and no retained model/engine/native code.
This is a measurement, not a before/after speed or size optimization claim.
Actual package export/declaration payload: 1,513 files; targets present PASS.
Native-absent imports, typed public native-load rejection without WASM fallback,
and strict public declaration compilation also PASS. Browser import-graph check
passes for 319 local modules reachable from 27 built entry points.

Retained fixture and full receipt:
`C:/Users/steam/AppData/Local/Temp/asrjs-packed-consumer-OsNeKt/compact-browser-bundle.json`.
Archive SHA256: `ab733fd07e28f0554c8b5890ef2e7754e24d1e340247cc19ce102b62ccf17ec2`.
Output SHA256s respectively:
`0d362c78cbb06667d57148c71bbde78dad69ac55ad7f9cf188ce677b1b2b3729` and
`22e6bdce40a128611ce121e064b8fba6b791abfc9769cf37c89df7812d27ec8a`.
The receipt hashes archive, package metadata, probe and loaded module bytes; the
temporary fixture is retained but may be cleaned by the host later. This report
keeps measured results and reproduction commands, not an exhaustive source map.

## Reproduce and checks

```powershell
npm run build
node --test tests/smoke/compact-browser-bundle.test.mjs
node tests/smoke/packed-native-absent-smoke.mjs
node tests/smoke/package-tarball-smoke.mjs
node tests/smoke/browser-root-import-graph.mjs
node node_modules/eslint/bin/eslint.js --no-ignore tests/smoke/compact-browser-bundle.mjs tests/smoke/compact-browser-bundle.test.mjs tests/smoke/packed-native-absent-smoke.mjs
```

Four updated/new test cases RED before implementation; final seven PASS, including
default source mode and compiled-path negatives. Syntax, formatting and explicit
non-ignored lint PASS. The packed script now imports its existing URL constructor
explicitly from node:url rather than depending on the global lint configuration.
Independent read-only review APPROVE. Windows sandbox denied esbuild access to
temporary fixture ancestors; that live process was stopped, not counted as PASS.
The identical tests and real archive run then passed under host permissions.
Final related Vitest checks also PASS: 20 tests across package-tarball contract,
browser Node stub and Node I/O smoke suites. This is not a full-suite rerun.

## Scope and remaining work

Two extracted-archive browser-target bundles are covered, not every public export,
browser startup/inference, npm installation size or engine memory. Dependencies
were linked from installed local packages. No registry download, publication,
global settings, model artifact or sibling source change occurred. Fresh dist
includes the preserved dirty current tree, not only committed changes.

C14 cancellation/unload still needs sibling source authority: live browser-demo
HEAD `3bdb9f11208c63d30f540e1d678e20082236e4a7` has only `vite.config.js` dirty.
App.jsx SHA256 `669601d87bf6c6277f9619235defb537aabc135d918ce72dc901ae0f26d76804`
has no Cancel/Unload action or transcription AbortSignal; local-model/file flow
was already covered by C55. Do not rerun an unchanged smoke to close that gap.
Sibling write approval was requested again; no changes were made there.
Orukeet numerical/quality and Redux original-runtime gates remain unchanged.
