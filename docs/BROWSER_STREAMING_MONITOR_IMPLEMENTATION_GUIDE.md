# Browser Streaming Monitor Implementation Guide

This guide records how the current browser streaming monitor is structured in `@asrjs/speech-recognition` and how apps such as `streaming-demo` should consume it.

It is the implementation memory for the shipped monitor stack, not a design brainstorm.

## 1. Design Rule

The browser monitor is split into:

- framework-agnostic runtime logic in `src/runtime`
- thin framework wrappers in apps

The runtime owns:

- rendering logic
- visual semantics
- display mode metadata
- monitor snapshot shaping

The app owns:

- component lifecycle
- local layout
- event handlers
- persistence of user-selected view state

## 2. Shared Runtime Modules

### 2.1 `browser-waveform.ts`

Owns the canvas renderer for the streaming waveform.

Current responsibilities:

- top TEN-VAD lane
- center waveform lane
- bottom pre-VAD RMS lane
- threshold overlays
- accepted segment boxes
- logical onset markers
- active segment highlighting
- segment annotation band above the waveform
- small debugging legend for waveform semantics

Current color semantics:

- green bars: current active segment
- dark blue bars: rough-gate detector pass
- muted blue bars: included in segment bounds but below the live gate
- gray bars: outside segments and below gate

Current segment semantics:

- box start: extracted segment start
- onset: logical backtracked start inside the extracted box
- live: currently open segment

### 2.2 `browser-compact-stats.ts`

Owns the compact summary strip renderer.

Public API:

- `createBrowserCompactStatsRenderer(container)`
- `resolveBrowserCompactStatsSnapshot(snapshot, liveNoiseDbfs, liveSpeechDbfs)`

Current compact strip focuses on:

- speaking indicator
- SNR bar
- signal and average dBFS
- gate and noise dBFS
- visible buffer span
- recent accepted and rejected counts

### 2.3 `browser-monitor-display.ts`

Owns shared browser monitor display metadata.

Current exports include:

- `BROWSER_WAVEFORM_SCALE_OPTIONS`
- `BROWSER_MONITOR_SURFACE_OPTIONS`
- `DEFAULT_BROWSER_WAVEFORM_SCALE_MODE`
- `DEFAULT_BROWSER_MONITOR_SURFACE_MODE`
- `BROWSER_MONITOR_DISPLAY_CONTROL_DEFINITIONS`
- `BROWSER_BINARY_TOGGLE_OPTIONS`
- `BROWSER_REVERSED_BINARY_TOGGLE_OPTIONS`

These definitions are the source of truth for labels such as:

- `Surface`
- `View`
- `Gate lines`
- `TEN line`
- `RMS`

Apps should not duplicate those strings unless they are intentionally diverging from the runtime.

## 3. View Modes

Current surface modes:

- `full`
- `stats-only`
- `waveform-only`

Current scale modes:

- `physical`
- `focus`
- `adaptive`

Current default scale mode is:

- `physical`

The default is owned by the runtime export and then consumed by apps when they initialize state.

## 4. Debugging Semantics

The current waveform is intended to make segmentation debugging observable.

Important distinctions:

- The live gate does not define final extracted bounds by itself.
- Backtracked onset can move the logical segment start earlier than the gate-open moment.
- Preroll can move the extracted start earlier than the logical onset.
- Included lead-in audio can therefore be inside a segment box while still being below the live gate.

That is why the current renderer distinguishes:

- extracted segment box
- logical onset marker
- included-but-below-gate waveform color
- live detector-pass waveform color

## 5. Segment Labeling

Current segment labels are intentionally compact because the visible window can contain many short segments.

Current label format in the demo wrapper:

- `sil:320ms`
- `max:640ms`
- `man:240ms`

Current abbreviations:

- `silence -> sil`
- `max-duration -> max`
- `manual -> man`
- `stop -> stop`

Do not use unstable running ids such as `1`, `2`, `3` in the visible labels. They renumber as the window scrolls and make the display harder to reason about.

## 6. Annotation Band Rules

The current renderer uses a dedicated segment annotation band above the waveform body.

Rules:

- labels should not be drawn directly over waveform bars if readability matters
- labels should cascade across multiple short rows
- onset markers may extend downward through the waveform because they indicate a boundary, not just a label

If label density becomes too high again, prefer this order of simplification:

1. shorten label text
2. drop duration for very dense views
3. increase cascade rows slightly
4. hide labels for segments below a minimum visible width

Do not go back to unstable numeric ids as the first fix.

## 7. App Integration Pattern

`streaming-demo` is the current example consumer.

The app wrapper should:

- subscribe to the shared monitor
- map runtime snapshot data into renderer-friendly payloads
- provide user state such as display mode and scale mode
- mount or unmount renderers to avoid wasted work

The app wrapper should not:

- re-implement renderer semantics
- duplicate monitor labels
- invent a second source of truth for display mode options

## 8. Files To Check When Updating The Monitor

If you change the streaming monitor, check these files together:

- `src/runtime/browser-waveform.ts`
- `src/runtime/browser-compact-stats.ts`
- `src/runtime/browser-monitor-display.ts`
- `src/browser.ts`
- `streaming-demo/src/BrowserConsoleLayout.jsx`
- `streaming-demo/src/shared/streaming/components/StreamingCapturePanel.jsx`
- `streaming-demo/src/shared/streaming/components/WaveformCanvas.jsx`
- `streaming-demo/src/shared/streaming/components/WaveformCompactStats.tsx`

## 9. Current Intent

The current browser monitor is not just decorative. It is a debugging surface for:

- noise floor adaptation
- rough-gate opening and closing
- onset backtracking
- extracted segment bounds
- final segment acceptance behavior

Any future redesign should preserve that observability, even if the visual style changes.
