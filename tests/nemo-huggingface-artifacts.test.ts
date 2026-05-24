import { describe, expect, it, vi } from 'vitest';

vi.mock('../src/runtime/huggingface.js', () => ({
  fetchModelFiles: vi.fn(),
}));

import { fetchModelFiles } from '../src/runtime/huggingface.js';
import { hasHuggingFaceExternalDataFile } from '../src/models/nemo-common/huggingface-artifacts.js';

const fetchModelFilesMock = vi.mocked(fetchModelFiles);

describe('NeMo Hugging Face artifact helpers', () => {
  it('detects external ONNX data files published next to model weights', async () => {
    fetchModelFilesMock.mockResolvedValueOnce([
      'encoder-model.onnx',
      'encoder-model.onnx.data',
      'decoder_joint-model.int8.onnx',
    ]);

    await expect(
      hasHuggingFaceExternalDataFile(
        'ysdede/parakeet-tdt-0.6b-v3-onnx',
        'feat/fp16-canonical-v3',
        'encoder-model.onnx',
      ),
    ).resolves.toBe(true);
  });

  it('does not invent external data for self-contained weights', async () => {
    fetchModelFilesMock.mockResolvedValueOnce([
      'encoder-model.fp16.onnx',
      'decoder_joint-model.int8.onnx',
    ]);

    await expect(
      hasHuggingFaceExternalDataFile(
        'ysdede/parakeet-tdt-0.6b-v3-onnx',
        'feat/fp16-canonical-v3',
        'encoder-model.fp16.onnx',
      ),
    ).resolves.toBe(false);
  });
});
