import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import { parseWhisperManifest } from '../src/models/whisper-seq2seq/manifest.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from '../src/models/whisper-seq2seq/types.js';

describe('Whisper 4-graph splitgraph fixture smoke', () => {
  it('loads splitgraph artifacts, runs encoder→init→step→align, verifies shapes', async () => {
    const fixtureDir = process.env.WHISPER_SPLITGRAPH_FIXTURE_DIR;
    if (!fixtureDir) {
      console.warn(
        'Skipping: set WHISPER_SPLITGRAPH_FIXTURE_DIR=/path/to/exported/whisper-tiny',
      );
      return;
    }

    const encoderPath = path.join(fixtureDir, 'encoder_model.onnx');
    const decoderInitPath = path.join(fixtureDir, 'decoder_init.onnx');
    const decoderStepPath = path.join(fixtureDir, 'decoder_step.onnx');
    const decoderAlignPath = path.join(fixtureDir, 'decoder_align.onnx');
    const tokenizerPath = path.join(fixtureDir, 'tokenizer.json');
    const manifestPath = path.join(fixtureDir, 'manifest.json');

    const requiredFiles = [
      encoderPath, decoderInitPath, decoderStepPath,
      tokenizerPath, manifestPath,
    ];
    if (!requiredFiles.every((f) => fs.existsSync(f))) {
      console.warn(
        `Skipping: missing splitgraph fixtures in ${fixtureDir}. ` +
        `Run: tools/whisper-onnx-export/.venv/bin/python export_whisper.py openai/whisper-tiny ${fixtureDir}`,
      );
      return;
    }

    // 1. Parse manifest to verify dimensions are config-driven
    const manifestRaw = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const parsed = parseWhisperManifest(manifestRaw);
    const modelConfig = parsed.modelConfig;
    const genConfig = parsed.generationConfig;

    // Verify manifest-driven dimensions (not hardcoded tiny)
    expect(modelConfig.decoderLayers).toBeGreaterThan(0);
    expect(modelConfig.decoderAttentionHeads).toBeGreaterThan(0);
    expect(modelConfig.dModel).toBeGreaterThan(0);
    expect(modelConfig.headDim).toBeGreaterThan(0);
    expect(modelConfig.dModel / modelConfig.decoderAttentionHeads).toBe(modelConfig.headDim);
    expect(genConfig.alignmentHeads.length).toBeGreaterThan(0);

    // 2. Build config from manifest
    const config: WhisperSeq2SeqModelConfig = {
      ecosystem: 'openai',
      architecture: 'whisper-seq2seq',
      melBins: (manifestRaw.num_mel_bins as number) ?? 80,
      sampleRate: 16000,
      maxSourcePositions: (manifestRaw.max_source_positions as number) ?? 3000,
      maxTargetPositions: (manifestRaw.max_target_positions as number) ?? 448,
      vocabularySize: (manifestRaw.vocab_size as number) ?? 51865,
      languages: ['tr', 'en'],
      processorArchitecture: 'whisper-mel',
      encoderArchitecture: 'whisper-transformer',
      decoderArchitecture: 'transformer-decoder',
      tokenizer: { kind: 'tiktoken', vocabSize: (manifestRaw.vocab_size as number) ?? 51865 },
    };

    // 3. Create splitgraph source
    const source: WhisperArtifactSource = {
      kind: 'splitgraph',
      artifacts: {
        encoderUrl: `file://${encoderPath}`,
        decoderInitUrl: `file://${decoderInitPath}`,
        decoderStepUrl: `file://${decoderStepPath}`,
        decoderAlignUrl: fs.existsSync(decoderAlignPath)
          ? `file://${decoderAlignPath}`
          : undefined,
        tokenizerUrl: `file://${tokenizerPath}`,
        manifestUrl: `file://${manifestPath}`,
      },
    };

    // 4. Create executor
    const executor = new WhisperOnnxExecutor(
      'whisper-tiny-self-export',
      { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
      config,
      'wasm',
      { source },
    );

    await executor.ready();

    try {
      // 5. Generate test audio: 2.5s of 440 Hz + 880 Hz chirp at 16 kHz
      const sampleRate = 16000;
      const duration = 2.5;
      const totalSamples = Math.floor(sampleRate * duration);
      const samples = new Float32Array(totalSamples);
      for (let i = 0; i < totalSamples; i++) {
        const t = i / sampleRate;
        // Frequency sweep from 440 to 880 Hz
        const freq = 440 + (440 * t) / duration;
        samples[i] = Math.sin(2 * Math.PI * freq * t) * 0.5;
      }

      const audio = {
        sampleRate,
        durationSeconds: duration,
        channels: [samples],
        numberOfChannels: 1,
        numberOfFrames: totalSamples,
      };

      // 6. Run transcription
      const result = await executor.transcribe(
        audio,
        {
          language: 'tr',
          maxNewTokens: 30,
          returnWords: true,
          returnTimestamps: 'word',
          returnSpecialTokens: false,
        },
        { modelId: 'whisper-tiny-self-export', config },
      );

      // 7. Verify: no stub warning
      const stubWarning = result.warnings?.find(
        (w) => w.code === 'whisper-seq2seq.stubbed-decoder',
      );
      expect(stubWarning).toBeUndefined();

      // 8. Verify: tokens produced (including EOS)
      expect(Array.isArray(result.tokens)).toBe(true);
      expect(result.tokens!.length).toBeGreaterThan(0);

      // 9. Verify: utterance text is present
      expect(typeof result.utteranceText).toBe('string');

      // 10. Verify: language set
      expect(result.language).toBe('tr');

      // 11. Verify: segments have reasonable time bounds
      if (result.segments && result.segments.length > 0) {
        for (const seg of result.segments) {
          expect(seg.startTime).toBeGreaterThanOrEqual(0);
          expect(seg.endTime).toBeGreaterThanOrEqual(seg.startTime);
        }
      }

      // 12. Verify word-level timestamps if alignment produced them
      if (result.words && result.words.length > 0) {
        expect(result.words.length).toBeGreaterThan(0);
        for (const word of result.words) {
          expect(word.startTime).toBeGreaterThanOrEqual(0);
          expect(word.endTime).toBeGreaterThanOrEqual(word.startTime);
          expect(typeof word.text).toBe('string');
        }

        // Verify word probabilities are in [0, 1] range (from DTW alignment)
        for (const word of result.words) {
          if (word.confidence !== undefined && word.confidence >= 0) {
            expect(word.confidence).toBeLessThanOrEqual(1);
          }
        }
      }

      console.log('Splitgraph smoke test passed:', {
        tokens: result.tokens?.length,
        utteranceText: result.utteranceText.substring(0, 60),
        segments: result.segments?.length,
        words: result.words?.length,
      });
    } finally {
      await executor.dispose();
    }
  }, 120000); // 2 min timeout for model download+inference
});
