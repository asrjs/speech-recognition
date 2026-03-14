# asr.js Benchmark Demo

This demo benchmarks `asr.js` Parakeet TDT models against Hugging Face speech datasets.

It is intentionally built on the current `asr.js` helper surface instead of carrying over the old `parakeet.js` app structure. The demo uses:

- built-in Parakeet model loading from `asr.js`
- dataset split and row helpers from `asr.js`
- benchmark statistics and CSV export helpers from `asr.js`
- shared browser audio decoding helpers from `asr.js`

## Run

```bash
npm install
npm run dev
```

The app aliases `asr.js` to the local repo source tree, so it always tests the current library state.

## What It Covers

- choose Parakeet model, backend, quantization, and preprocessor mode
- browse Hugging Face dataset configs and splits
- prepare sequential or seeded-random benchmark sample sets
- run repeat benchmarks against real dataset audio
- inspect timing, repeatability, and transcript output
- export run records as JSON or CSV

## Notes

- This demo is focused on the `asr.js` direct model/runtime surface, not old compatibility APIs.
- It keeps the UI smaller than the copied benchmark app on purpose. The reusable pieces belong in `asr.js`; the demo should stay readable.
- The shared audio helpers are also suitable for other browser apps that need to fetch and decode audio before inference.
