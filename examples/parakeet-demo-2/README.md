# asr.js Parakeet Inspector Demo

This demo is a rebuilt `asr.js` example for Parakeet TDT models.

It is intentionally not a `transformers.js` pipeline demo. The app uses the current `asr.js` model-loading and direct transcript surface instead:

- load Parakeet TDT from Hugging Face or a local folder
- choose backend, quantization, and preprocessor settings
- run direct transcription with native Parakeet output
- inspect metrics, tokens, timings, and repeat-run benchmark behavior

## Run

```bash
npm install
npm run dev
```

The demo aliases `asr.js` to the local repo source tree, so it always exercises the current library state.

## Notes

- This demo is focused on the `asr.js` runtime surface, not compatibility with external pipeline APIs.
- Local folder loading uses the shared Parakeet helpers exported by `asr.js`.
- The bundled sample audio is served from `public/assets/Harvard-L2-1.ogg`.
