/** A pinned ONNX bundle published for direct loading from the Hugging Face Hub. */
export interface PublishedBundle {
  readonly repoId: string;
  readonly revision: string;
  readonly files: Readonly<Record<string, string>>;
}

/**
 * Build a stable Hub URL for a known artifact. Revisions are immutable commit
 * ids so a catalog entry cannot silently change underneath a cached model.
 */
export function publishedBundleFileUrl(bundle: PublishedBundle, fileKey: string): string {
  const path = bundle.files[fileKey];
  if (!path)
    throw new Error(`Published bundle ${bundle.repoId} has no artifact named "${fileKey}".`);
  const repo = bundle.repoId.split('/').map(encodeURIComponent).join('/');
  const revision = encodeURIComponent(bundle.revision);
  const encodedPath = path.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repo}/resolve/${revision}/${encodedPath}`;
}

export const QWEN3_ASR_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/qwen3-asr-0.6b-onnx',
  revision: 'ac0bedcc0f1614c0b3982940c14bdb169ff2365d',
  files: {
    encoder: 'audio-encoder-dynamic.onnx',
    decoderPrefill: 'decoder-prefill-fp16.onnx',
    decoderStep: 'decoder-step-fp16.onnx',
    decoderData: 'decoder-fp16.onnx.data',
    tokenizer: 'tokenizer/tokenizer.json',
  },
};

export const MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/moonshine-streaming-tiny-onnx',
  revision: 'cb347a0760c2d7d6cf9083eb770376ba108e5b31',
  files: {
    encoder: 'encoder_model.onnx',
    encoderData: 'encoder_model.onnx.data',
    chunkedEncoder: 'encoder_chunked_model.onnx',
    chunkedEncoderData: 'encoder_chunked_model.onnx.data',
    decoderPrefill: 'decoder_prefill_model.onnx',
    decoderPrefillData: 'decoder_prefill_model.onnx.data',
    decoderStep: 'decoder_step_model.onnx',
    decoderStepData: 'decoder_step_model.onnx.data',
    tokenizer: 'tokenizer.json',
    generation: 'generation_config.json',
    config: 'config.json',
    preprocessor: 'preprocessor_config.json',
    processor: 'processor_config.json',
    tokenizerConfig: 'tokenizer_config.json',
  },
};

export const GIGAAM_CTC_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/gigaam-multilingual-ctc-onnx',
  revision: 'd97e0c3a024eba7a167bace280de827aa970506b',
  files: { model: 'multilingual_ctc.onnx', tokenizer: 'multilingual_vocab.txt' },
};

export const GIGAAM_RNNT_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/gigaam-v3-e2e-rnnt-onnx',
  revision: '0649dfcb6271a713c5aec4321d97d52f3988cd30',
  files: {
    encoder: 'v3_e2e_rnnt_encoder.onnx',
    decoder: 'v3_e2e_rnnt_decoder.onnx',
    joint: 'v3_e2e_rnnt_joint.onnx',
    tokenizer: 'v3_e2e_rnnt_vocab.txt',
  },
};

export const SENSEVOICE_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/sensevoice-small-onnx',
  revision: 'b6c40945d2759130ea312d14f92ddad079ba7933',
  files: { model: 'model.onnx', tokenizer: 'vocab.txt', cmvn: 'am.mvn' },
};

export const XASR_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/x-asr-zh-en-160ms-onnx',
  revision: '19e639c536041f60aface640c6e6a944fc3915c9',
  files: {
    encoder: 'encoder-160ms.onnx',
    decoder: 'decoder-160ms.onnx',
    joiner: 'joiner-160ms.onnx',
    tokenizer: 'tokens.txt',
  },
};

export const PARAKEET_CTC_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/parakeet-hf-ctc-110m-en-onnx',
  revision: '27d8c28aba465e4afee5b85f9f8073d9a6735758',
  files: { model: 'model.onnx', tokenizer: 'tokenizer.json' },
};

export const WAV2VEC2_TURKISH_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/wav2vec2-xls-r-300m-cv7-turkish-onnx',
  revision: '44e8fcf6b23dce1e39516c0d90254d47d2a6f33b',
  files: { model: 'model.onnx', tokenizer: 'vocab.json' },
};

export const OMNILINGUAL_CTC_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/omnilingual-asr-ctc-300m-onnx',
  revision: 'ce2fafb218ca0074880def186a25abbf095af19e',
  files: { model: 'model.int8.onnx', tokenizer: 'tokens.txt' },
};

export const MOONSHINE_TINY_PUBLISHED_BUNDLE: PublishedBundle = {
  repoId: 'ysdede/moonshine-tiny-onnx',
  revision: '4cea222534a2151ec3f193f4d5ea888d9455f151',
  files: {
    encoder: 'encoder_model.onnx',
    encoderData: 'encoder_model.onnx.data',
    crossKv: 'decoder_cross_kv_model.onnx',
    crossKvData: 'decoder_cross_kv_model.onnx.data',
    decoder: 'decoder_with_past_model.onnx',
    decoderData: 'decoder_with_past_model.onnx.data',
    tokenizer: 'tokenizer.json',
    generation: 'generation_config.json',
    receipt: 'export-receipt.json',
  },
};
