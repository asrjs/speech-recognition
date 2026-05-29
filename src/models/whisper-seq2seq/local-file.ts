/**
 * Self-exported 4-graph Whisper local-file loader.
 *
 * Reads a directory containing the whisper-browser-self-export-v1 format:
 *   manifest.json, encoder_model.onnx, decoder_init.onnx,
 *   decoder_step.onnx, decoder_align.onnx, tokenizer files
 *
 * Returns a WhisperArtifactSource + WhisperSeq2SeqModelConfig ready for
 * createWhisperSeq2SeqModelFamily().createModel().
 */

import * as fs from 'fs';
import * as path from 'path';
import { parseWhisperManifest } from './manifest.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from './types.js';

export interface SplitGraphLocalModel {
  readonly source: WhisperArtifactSource;
  readonly config: WhisperSeq2SeqModelConfig;
  readonly modelId: string;
}

export function loadSplitGraphLocalModel(dirPath: string): SplitGraphLocalModel {
  const resolved = path.resolve(dirPath);

  const manifestPath = path.join(resolved, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`manifest.json not found in ${resolved}. Is this a self-exported Whisper directory?`);
  }

  const manifestRaw = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
  const manifest = parseWhisperManifest(manifestRaw);

  const modelId = manifest.modelId || path.basename(resolved);

  const fileUrl = (name: string): string => {
    const fullPath = path.join(resolved, name);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing artifact: ${name} in ${resolved}`);
    }
    return `file://${fullPath}`;
  };

  const config: WhisperSeq2SeqModelConfig = {
    ecosystem: 'openai',
    architecture: 'whisper-seq2seq',
    processorArchitecture: 'whisper-mel',
    encoderArchitecture: 'whisper-transformer',
    decoderArchitecture: 'transformer-decoder',
    sampleRate: 16000,
    melBins: (manifestRaw.num_mel_bins as number) ?? 80,
    maxSourcePositions: (manifestRaw.max_source_positions as number) ?? 3000,
    maxTargetPositions: (manifestRaw.max_target_positions as number) ?? 448,
    vocabularySize: (manifestRaw.vocab_size as number) ?? 51865,
    languages: ['en'],
    tokenizer: {
      kind: 'tiktoken',
      vocabSize: (manifestRaw.vocab_size as number) ?? 51865,
    },
  };

  const source: WhisperArtifactSource = {
    kind: 'splitgraph',
    artifacts: {
      encoderUrl: fileUrl('encoder_model.onnx'),
      decoderInitUrl: fileUrl('decoder_init.onnx'),
      decoderStepUrl: fileUrl('decoder_step.onnx'),
      decoderAlignUrl: fs.existsSync(path.join(resolved, 'decoder_align.onnx'))
        ? fileUrl('decoder_align.onnx')
        : undefined,
      tokenizerUrl: fileUrl('tokenizer.json'),
      manifestUrl: fileUrl('manifest.json'),
    },
  };

  return { source, config, modelId };
}
