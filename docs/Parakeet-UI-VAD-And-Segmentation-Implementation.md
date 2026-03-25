# Parakeet-UI VAD And Segmentation Implementation

This document describes how `parakeet-ui` actually detects speech, tracks background noise, creates segments, and discards non-speech.

It is based on the current implementation in:

- `src/audio-processor.js`
- `src/AudioManager.js`
- `src/AudioSegmentProcessor.js`
- `src/config/audioParams.js`
- `src/utils/ringBuffer.js`

This is the implementation truth for `parakeet-ui`. It is not a generic DSP note and it is not the `asrjs` runtime.

## 1. High-Level Flow

The project uses a four-stage pipeline:

1. `AudioWorklet` accumulates microphone samples into fixed analysis windows.
2. The worklet computes a smoothed peak-amplitude energy value and posts audio windows to the main thread.
3. `AudioSegmentProcessor` uses that window energy plus an adaptive noise floor and heuristic SNR to decide when speech starts and ends.
4. `AudioManager.handleNewSegment()` extracts PCM from the ring buffer and runs a second validation gate using duration and 16 kHz-normalized energy thresholds.

So there are two different decisions:

- live VAD boundary detection
- final segment acceptance/rejection

## 2. Audio Capture

### 2.1 Worklet analysis window

`src/audio-processor.js` uses:

```js
const windowDuration = 0.080;
this.bufferSize = Math.round(windowDuration * sampleRate);
```

So the detector processes audio in **80 ms windows**.

Examples:

- `16 kHz -> 1280 samples`
- `22.05 kHz -> 1764 samples`
- `48 kHz -> 3840 samples`

The worklet accumulates raw microphone samples until the window is full, then posts:

- `audioData`: a copy of the 80 ms PCM window
- `energy`: the worklet's smoothed energy metric
- `sampleRate`
- `timestamp`

### 2.2 Ring buffer

`AudioManager` stores incoming PCM in `RingBuffer`, not by concatenating arrays.

Key properties:

- fixed duration: `120 seconds`
- global frame addressing
- monotonic time from `ringBuffer.getCurrentTime()`
- old audio is overwritten automatically

Segment extraction later uses exact sample ranges from this ring buffer.

## 3. The Worklet Energy Metric

This project does **not** use RMS in the live worklet detector.

The worklet computes:

```text
maxAbsValue = max(abs(sample))
energy = SMA(maxAbsValue, last 6 windows)
```

In code:

```js
const maxAbsValue = Math.max(...Array.from(buffer).map(Math.abs));
this.maxValues.push(maxAbsValue);
const energy = sum(this.maxValues) / this.maxValues.length;
```

This means the live detector uses:

- peak amplitude per 80 ms window
- then a 6-window simple moving average

With 80 ms windows and `smaLength = 6`, the smoothing span is about:

```text
6 * 80 ms = 480 ms
```

This is important because later formulas refer to "energy", but in the live detector that value is really **smoothed peak amplitude**, not RMS power.

## 4. Live Speech Decision In `AudioSegmentProcessor`

`AudioSegmentProcessor.processAudioData()` receives:

- `chunk`
- `currentTime`
- `energy`

and immediately computes:

```js
const isSpeech = energy > this.options.energyThreshold;
```

So the primary live speech trigger is still:

```text
isSpeech = smoothedPeakEnergy > audioThreshold
```

Defaults from `audioParams.js`:

- `audioThreshold = 0.08`
- `windowDuration = 0.080`
- `smaLength = 6`

## 5. Background Noise Tracking

The processor maintains a rolling `noiseFloor`.

Initial state:

```js
noiseFloor: 0.005
```

### 5.1 When the noise floor updates

The noise floor only updates when:

```text
isSpeech == false
```

So the baseline is learned from windows currently classified as non-speech.

### 5.2 Adaptation formula

When the current window is non-speech:

```text
noiseFloor_next =
  noiseFloor_prev * (1 - adaptationRate) +
  energy * adaptationRate
```

with:

```text
noiseFloor >= 0.00001
```

### 5.3 Dynamic adaptation rate

The code uses faster adaptation early in a silence run, then slows down.

If `silenceDuration < minBackgroundDuration`, then:

```text
blendFactor = min(1, silenceDuration / minBackgroundDuration)
adaptationRate =
  fastAdaptationRate * (1 - blendFactor) +
  noiseFloorAdaptationRate * blendFactor
```

Otherwise:

```text
adaptationRate = noiseFloorAdaptationRate
```

Defaults:

- `noiseFloorAdaptationRate = 0.05`
- `fastAdaptationRate = 0.15`
- `minBackgroundDuration = 1.0`

Interpretation:

- first part of a silence adapts fast
- sustained silence adapts slowly and more stably

## 6. SNR Calculation

The processor computes:

```js
const snrDb = 10 * Math.log10(energy / noiseFloor);
```

So the live SNR is:

```text
SNR_dB = 10 * log10(smoothedPeakEnergy / noiseFloor)
```

This is a heuristic ratio built from the same peak-based energy metric. It is useful inside this detector, but it is not a strict power-spectrum SNR.

Defaults:

- `snrThreshold = 3.0`
- `minSnrThreshold = 1.0`

## 7. What SNR Is Actually Used For

The current processor does **not** start speech from SNR directly.

The code path is:

```text
live start trigger = energy > audioThreshold
```

SNR is then used for:

- diagnostics and stats
- onset backtracking
- deciding how far back to search for the true speech start

So the real behavior is:

- threshold-first onset
- SNR-assisted start refinement

not:

- SNR-first onset gating

## 8. Speech Start Logic

When the processor is idle and a window is classified as speech:

```js
if (!this.state.inSpeech && isSpeech) {
  let realStartIndex = this.findSpeechStart();
  let realStartTime = ...
  this.startSpeech(realStartTime, energy);
}
```

### 8.1 `findSpeechStart()`

The processor looks back through `recentChunks` and tries to recover an earlier onset.

Each stored chunk contains:

- `time`
- `energy`
- `isSpeech`
- `snr`

The algorithm:

1. Find the most recent chunk already marked as speech.
2. Walk backward looking for a rising energy trend:

```text
next.energy > current.energy * (1 + energyRiseThreshold)
```

3. Stop if SNR drops too far:

```text
snr < minSnrThreshold / 2
```

4. Stop after more than 6 chunks of lookback.
5. If no rising trend is found, look for the last point where:

```text
snr < minSnrThreshold
```

and return the next chunk.
6. If nothing is found, default to:

```text
firstSpeechIndex - 4 chunks
```

Defaults:

- `energyRiseThreshold = 0.08`
- max coded lookback in function: `6 chunks`
- default fallback lookback: `4 chunks`

Because chunks are 80 ms, that corresponds roughly to:

- max search window: about `480 ms`
- fallback lookback: about `320 ms`

### 8.2 Speech state activation

Once the start time is chosen:

```js
this.state.inSpeech = true;
this.state.speechStartTime = time;
this.state.silenceCounter = 0;
this.state.speechEnergies = [energy];
```

No separate minimum-speech confirmation happens here.

## 9. Speech End Logic

While in speech, if a window is non-speech:

```js
this.state.silenceCounter++;
if (this.state.silenceCounter >= (this.options.silenceThreshold * 10)) {
  finalize
}
```

This means the code converts `silenceThreshold` to windows by multiplying by `10`.

Implementation note:

- the detector windows are `80 ms`
- but this conversion behaves like legacy `100 ms` windows

So the actual silence timing is:

```text
actual_end_hold_seconds ≈ silenceLength * 10 * 0.08
                         ≈ silenceLength * 0.8
```

Examples with defaults/presets:

- `0.4 -> 4 windows -> about 320 ms`
- `1.0 -> 10 windows -> about 800 ms`

So comments in config that map these to exact window counts are not fully aligned with current 80 ms processing. The implementation still works, but the effective silence timing is shorter than the plain seconds value suggests.

## 10. Segment Splitting

The processor also proactively splits long speech.

If already in speech:

```js
currentSpeechDuration = currentTime - speechStartTime;
if (currentSpeechDuration > maxSegmentDuration) {
  createSegment(start, currentTime);
  startSpeech(currentTime, energy);
}
```

Default:

- `maxSegmentDuration = 4.8 seconds`

So one long utterance is broken into consecutive segments without waiting for silence.

## 11. What `createSegment()` Actually Returns

`AudioSegmentProcessor.createSegment()` no longer slices audio itself.

It returns only metadata:

- `startTime`
- `endTime`
- `duration`

The real PCM extraction happens later in `AudioManager.handleNewSegment()`.

## 12. Segment Refinement In `AudioManager`

When `AudioManager` receives a segment from the processor, it modifies timing before extracting audio.

### 12.1 Fixed lookback

First it extends the start backward:

```js
segment.startTime = Math.max(0, segment.startTime - lookbackDuration);
```

Default:

- `lookbackDuration = 0.120 seconds`

This is applied **after** the processor already used `findSpeechStart()`.

So there are two separate early-start protections:

1. chunk-history onset backtracking in `AudioSegmentProcessor`
2. fixed 120 ms rewind in `AudioManager`

### 12.2 Inter-segment overlap

If there is a previous accepted segment and the new segment starts very close to it:

```js
if (timeSinceLastSegment < overlapDuration) {
  segment.startTime = Math.max(
    lastSegment.endTime - overlapDuration,
    segment.startTime - overlapDuration
  );
}
```

Default:

- `overlapDuration = 0.080 seconds`

So accepted segments can intentionally overlap by up to 80 ms.

### 12.3 End padding / hangover

For extraction, the end is padded:

```js
paddedEndSample =
  speechEndSample +
  Math.round(speechHangover * sampleRate)
```

Default:

- `speechHangover = 0.16 seconds`

This padding affects extracted PCM, not the logical segment end time.

## 13. Final PCM Extraction

The segment PCM comes from the ring buffer:

```js
segment.audioData = ringBuffer.read(paddedStartSample, paddedEndSample);
```

with:

- `paddedStartSample = max(0, speechStartSample)`
- `paddedEndSample = min(currentFrame, speechEndSample + hangover)`

This ensures sample-accurate extraction based on the current global frame position.

## 14. Final Acceptance Gate

The most important non-speech rejection happens in `AudioManager.handleNewSegment()`.

This gate is independent from the live `isSpeech` flag.

The segment is accepted only if all of these hold:

```text
duration >= minSpeechDuration
normalizedPowerAt16k >= minEnergyPerSecondThreshold
normalizedEnergyIntegralAt16k >= minEnergyIntegralThreshold
```

### 14.1 Segment power calculation

`getSegmentEnergyMetrics()` computes:

```text
sumOfSquares = Σ x[i]^2
averagePower = sumOfSquares / numSamples
duration = numSamples / sampleRate
```

This `averagePower` is the segment's per-sample mean square power.

### 14.2 16 kHz normalization

The code then normalizes this power to a 16 kHz equivalent:

```text
normalizedPowerAt16k = averagePower * 16000
normalizedEnergyIntegralAt16k = normalizedPowerAt16k * duration
```

Interpretation:

- `normalizedPowerAt16k` is the segment strength metric
- `normalizedEnergyIntegralAt16k` is total normalized energy over the segment duration

### 14.3 Default acceptance thresholds

Defaults:

- `minSpeechDuration = 0.240 seconds`
- `minEnergyPerSecond = 5`
- `minEnergyIntegral = 22`

So even if live VAD starts and ends a segment, it is still rejected if it is:

- too short
- too weak on average
- too weak in total accumulated energy

## 15. Adaptive Final Thresholds

If `useAdaptiveEnergyThresholds` is enabled, `AudioManager` raises or lowers the final gate based on current background level.

Default:

- `useAdaptiveEnergyThresholds = true`

### 15.1 Formula used

The code uses the processor's current `noiseFloor`, which is the live chunk-level peak-based energy metric, then transforms it into an approximate per-sample 16 kHz reference:

```text
normalizedNoiseFloor = noiseFloor / windowSize
noiseFloorAt16k = normalizedNoiseFloor * 16000
```

Then:

```text
adaptiveMinEnergyIntegral = noiseFloorAt16k * adaptiveEnergyIntegralFactor
adaptiveMinEnergyPerSecond = noiseFloorAt16k * adaptiveEnergyPerSecondFactor
```

with floors:

```text
minEnergyIntegralThreshold =
  max(minAdaptiveEnergyIntegral, adaptiveMinEnergyIntegral)

minEnergyPerSecondThreshold =
  max(minAdaptiveEnergyPerSecond, adaptiveMinEnergyPerSecond)
```

Defaults:

- `adaptiveEnergyIntegralFactor = 25.0`
- `adaptiveEnergyPerSecondFactor = 10.0`
- `minAdaptiveEnergyIntegral = 3`
- `minAdaptiveEnergyPerSecond = 1`

Implementation note:

- the segment gate uses mean-square power
- the adaptive baseline comes from the processor's smoothed peak-amplitude noise floor

So the adaptive gate is practical and useful, but not dimensionally pure DSP. It is a heuristic bridge between the live detector metric and the final segment metric.

## 16. How Background Noise Is Rejected

The system rejects background noise through several layers:

### 16.1 Peak-threshold live detector

Only windows above the smoothed peak threshold open speech.

### 16.2 Silence-only noise floor adaptation

Background baseline is updated only during windows currently considered non-speech.

### 16.3 SNR-guided onset search

If energy crosses threshold, the code still looks backward and avoids going too far into clearly lower-SNR background.

### 16.4 Duration gate

Very short bursts are rejected at final acceptance:

```text
duration < 0.240 s -> reject
```

### 16.5 Average power gate

Segments that are too weak overall are rejected:

```text
normalizedPowerAt16k < threshold -> reject
```

### 16.6 Energy integral gate

Segments with too little total accumulated energy are rejected:

```text
normalizedEnergyIntegralAt16k < threshold -> reject
```

### 16.7 Adaptive gate scaling

In noisier environments, final thresholds rise with the current noise-floor estimate.

## 17. Defaults From `audioParams.js`

Current important defaults:

| Parameter | Default |
| --- | --- |
| `windowDuration` | `0.080 s` |
| `smaLength` | `6` |
| `audioThreshold` | `0.08` |
| `silenceLength` | `0.4 s` |
| `speechHangover` | `0.16 s` |
| `lookbackDuration` | `0.12 s` |
| `overlapDuration` | `0.08 s` |
| `minSpeechDuration` | `0.24 s` |
| `minEnergyIntegral` | `22` |
| `minEnergyPerSecond` | `5` |
| `snrThreshold` | `3.0 dB` |
| `minSnrThreshold` | `1.0 dB` |
| `noiseFloorAdaptationRate` | `0.05` |
| `fastAdaptationRate` | `0.15` |
| `minBackgroundDuration` | `1.0 s` |
| `energyRiseThreshold` | `0.08` |
| `maxSegmentDuration` | `4.8 s` |
| `useAdaptiveEnergyThresholds` | `true` |

Segmentation presets:

| Preset | `audioThreshold` | `silenceLength` | `speechHangover` |
| --- | --- | --- | --- |
| fast | `0.120` | `0.1 s` | `0.08 s` |
| medium | `0.080` | `0.4 s` | `0.16 s` |
| slow | `0.060` | `1.0 s` | `0.24 s` |

## 18. Important Implementation Notes

These are worth keeping in mind when changing or porting the detector:

### 18.1 Live detector energy is peak-based, not RMS-based

The worklet computes smoothed max amplitude, not RMS.

### 18.2 `minSpeechDuration` is a final filter, not a live onset confirmer

Speech starts as soon as a window crosses threshold. The duration check happens only later in `AudioManager`.

### 18.3 Silence release timing is approximate

The code still uses:

```js
silenceThreshold * 10
```

even though windows are 80 ms, so the effective release time is shorter than the nominal seconds value.

### 18.4 Start protection happens twice

The system uses:

- chunk-history onset backtracking
- fixed lookback rewind in `AudioManager`

### 18.5 End protection is extraction-only

`speechHangover` pads the extracted audio tail, but it does not delay the live logical segment end.

## 19. Practical Summary

`parakeet-ui` currently works like this:

1. Capture 80 ms PCM windows in the worklet.
2. Compute `energy = SMA(max(abs(samples)))`.
3. Mark a window as speech when `energy > audioThreshold`.
4. Update noise floor only during non-speech windows.
5. Compute heuristic SNR from `10 * log10(energy / noiseFloor)`.
6. When speech starts, backtrack through recent windows using rising energy and SNR clues.
7. When silence persists long enough, finalize a segment.
8. Apply lookback, overlap, and hangover during extraction.
9. Compute segment mean-square power.
10. Normalize to 16 kHz and reject segments that are too short, too weak per second, or too weak in total energy.

That combination is why the system can:

- adapt to room noise
- recover clipped starts better than a plain threshold
- avoid keeping many short or weak noise bursts
- split long continuous speech into manageable segments
