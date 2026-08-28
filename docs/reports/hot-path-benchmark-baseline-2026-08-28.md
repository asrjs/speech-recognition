# ASR.js hot-path benchmark baseline

- Date: 2026-08-28
- Host: Windows x64, Node v26.2.0
- Runs: 50 timed iterations per scenario, after five warm-up calls

Run the baseline with:

```text
npm run benchmark:hot-paths -- --runs=50 --json
```

The command builds the package first, runs correctness checks, and reports
mean, p50, p90, minimum, and maximum wall-clock time for each scenario. To
evaluate a candidate optimization, run the same command on the base and
candidate commits with the same Node version, hardware, and run count. This
synthetic harness is a comparison tool; it is not a model-quality or browser
WebGPU benchmark.

## Baseline results

| Scenario                     | Fixture                                          | Mean (ms) | P50 (ms) | P90 (ms) |
| ---------------------------- | ------------------------------------------------ | --------: | -------: | -------: |
| `transcript-normalization`   | 24 repeated Turkish sentences                    |     0.070 |    0.056 |    0.076 |
| `streaming-transcript-merge` | 48 windows, 4 segments, 12 words, 48 tokens each |     0.264 |    0.226 |    0.388 |
| `audio-stereo-downmix`       | 2-channel, 2-second Float32 PCM                  |     0.182 |    0.167 |    0.230 |
| `inference-argmax`           | 4096-value Float32 logits                        |     0.009 |    0.009 |    0.010 |
| `inference-token-quality`    | 4096-value Float32 logits with entropy           |     0.205 |    0.184 |    0.279 |
| `whisper-logit-processor`    | 4096-value logits and timestamp constraints      |     0.031 |    0.038 |    0.052 |

These values are local baseline observations, not release targets. A hot-path
patch should include a new base/candidate capture and a correctness result
before it is accepted.

## Correctness coverage

The harness fails fast if any of these invariants regress:

- empty transcript merge and absent token alignment remain valid;
- out-of-range `argmax` reads do not produce an invalid index;
- all-`NaN` logits use the defined invalid-quality fallback;
- stereo downmix preserves the expected average;
- one-frame audio slices remain valid;
- Whisper suppression ignores token IDs outside the logits array.
