
## 2025-03-17 - [Buffer Pooling in MedAsrJsPreprocessor]
**Learning:** Instantiating new \`Float32Array\` and \`Float64Array\` inside the \`computeRawMel\` hot loop causes measurable performance degradation and memory spikes from garbage collection overhead, particularly when continuously processing audio.
**Action:** Always utilize class-level buffered arrays (e.g. \`this.emphasizedBuffer\`, \`this.paddedBuffer\`, \`this.rawMelBuffer\`) within hot signal processing loops to pre-allocate and dynamically resize buffers instead of recreating new instances.
