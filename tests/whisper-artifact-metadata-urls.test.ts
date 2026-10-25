import { afterEach, describe, expect, it, vi } from 'vitest';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import { resolveWhisperArtifacts } from '../src/models/whisper-seq2seq/ort.js';
import type { WhisperSplitGraphArtifactSource } from '../src/models/whisper-seq2seq/types.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

function createExecutor(): WhisperOnnxExecutor {
  return new WhisperOnnxExecutor(
    'whisper-metadata-url-test',
    { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
    {
      ecosystem: 'openai',
      architecture: 'whisper-seq2seq',
      processorArchitecture: 'whisper-mel',
      encoderArchitecture: 'whisper-transformer',
      decoderArchitecture: 'transformer-decoder',
      sampleRate: 16000,
      melBins: 80,
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      vocabularySize: 51865,
      languages: ['en'],
      tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
    },
    'wasm',
    undefined,
  );
}

describe('Whisper artifact metadata URLs', () => {
  it('preserves explicit metadata URLs through splitgraph resolution', () => {
    const source: WhisperSplitGraphArtifactSource = {
      kind: 'splitgraph',
      artifacts: {
        encoderUrl: 'blob:encoder',
        decoderInitUrl: 'blob:decoder-init',
        decoderStepUrl: 'blob:decoder-step',
        tokenizerUrl: 'blob:tokenizer',
        manifestUrl: 'blob:manifest',
        configUrl: 'blob:model-config',
        generationConfigUrl: 'blob:generation-config',
      },
    };

    const resolved = resolveWhisperArtifacts(source, 'wasm');

    expect(resolved.artifacts).toMatchObject({
      configUrl: 'blob:model-config',
      generationConfigUrl: 'blob:generation-config',
    });
  });

  it('loads metadata from explicit opaque URLs instead of deriving them from the tokenizer URL', async () => {
    const bodies = new Map<string, Record<string, unknown>>([
      ['blob:model-config', {
        decoder_layers: 32,
        decoder_attention_heads: 20,
        d_model: 1280,
        num_mel_bins: 128,
      }],
      ['blob:generation-config', {
        alignment_heads: [[2, 3]],
        no_timestamps_token_id: 50363,
      }],
    ]);
    const requested: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requested.push(url);
      const body = bodies.get(url);
      return body
        ? new Response(JSON.stringify(body), { status: 200 })
        : new Response('', { status: 404 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const executor = createExecutor() as unknown as {
      loadGenerationConfig(artifacts: {
        readonly tokenizerUrl: string;
        readonly generationConfigUrl?: string;
      }): Promise<{ readonly alignmentHeads: readonly { readonly layer: number; readonly head: number }[] }>;
      loadModelConfig(artifacts: {
        readonly tokenizerUrl: string;
        readonly configUrl?: string;
        readonly generationConfigUrl?: string;
      }): Promise<{ readonly decoderLayers: number; readonly dModel: number; readonly numMelBins?: number }>;
    };
    const artifacts = {
      tokenizerUrl: 'blob:opaque-tokenizer-locator',
      configUrl: 'blob:model-config',
      generationConfigUrl: 'blob:generation-config',
    };

    const generationConfig = await executor.loadGenerationConfig(artifacts);
    const modelConfig = await executor.loadModelConfig(artifacts);

    expect(requested).toEqual(['blob:generation-config', 'blob:model-config']);
    expect(generationConfig.alignmentHeads).toEqual([{ layer: 2, head: 3 }]);
    expect(modelConfig).toMatchObject({ decoderLayers: 32, dModel: 1280, numMelBins: 128 });
  });

  it('retains tokenizer-sibling URL fallback for existing remote sources', async () => {
    const requested: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requested.push(url);
      const body = url.endsWith('/config.json')
        ? { decoder_layers: 6, decoder_attention_heads: 8, d_model: 512, num_mel_bins: 80 }
        : { alignment_heads: [[1, 2]] };
      return new Response(JSON.stringify(body), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const executor = createExecutor() as unknown as {
      loadGenerationConfig(artifacts: { readonly tokenizerUrl: string }): Promise<unknown>;
      loadModelConfig(artifacts: { readonly tokenizerUrl: string }): Promise<unknown>;
    };
    const artifacts = { tokenizerUrl: 'https://models.example/whisper/tokenizer.json' };

    await executor.loadModelConfig(artifacts);
    await executor.loadGenerationConfig(artifacts);

    expect(requested).toEqual([
      'https://models.example/whisper/config.json',
      'https://models.example/whisper/generation_config.json',
    ]);
  });
});
