# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2018 Ryan Leary
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Explicit eval-only NeMo real-pair STFT export adapter.

Arithmetic follows NeMo FilterbankFeatures (NVIDIA, Apache-2.0); official
normalization and length helpers are reused by default. An explicit diagnostic
normalizer can be injected for isolated export experiments. No installed
source monkeypatch. This candidate must pass a captured oracle before use.
"""
import torch


class RealPairProcessor(torch.nn.Module):
    def __init__(self, processor, *, include_stages=False, diagnostic_normalizer=None):
        super().__init__()
        f = processor.featurizer
        if f.training or f.use_grads or f.exact_pad or f.stft_pad_amount is not None:
            raise ValueError('Real-pair export requires eval, no grads and centered STFT')
        if f.frame_splicing != 1 or f.pad_to != 0 or f.normalize != 'per_feature':
            raise ValueError('Real-pair export requires no splicing/padding and per_feature normalization')
        self.featurizer = f
        self.include_stages = include_stages
        if diagnostic_normalizer is not None and not isinstance(diagnostic_normalizer, torch.nn.Module):
            raise TypeError('Diagnostic normalizer must be a torch module')
        self.diagnostic_normalizer = diagnostic_normalizer

    def forward(self, waveform, length):
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch
        f = self.featurizer
        unfixed = f.get_seq_len(length)
        valid = torch.where(length == 0, torch.zeros_like(unfixed), unfixed)
        x = waveform
        if f.preemph is not None:
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < length.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - f.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~mask, 0.0)
        pairs = torch.stft(x, n_fft=f.n_fft, hop_length=f.hop_length,
            win_length=f.win_length, center=True, window=f.window.to(dtype=torch.float, device=x.device),
            return_complex=False, pad_mode='constant')
        x = torch.sqrt(pairs.pow(2).sum(-1))
        if f.mag_power != 1.0:
            x = x.pow(f.mag_power)
        power = x
        x = torch.matmul(f.fb.to(x.dtype), x)
        mel = x
        if f.log:
            if f.log_zero_guard_type == 'add':
                x = torch.log(x + f.log_zero_guard_value_fn(x))
            elif f.log_zero_guard_type == 'clamp':
                x = torch.log(torch.clamp(x, min=f.log_zero_guard_value_fn(x)))
            else:
                raise ValueError('Unsupported log guard')
        log_mel = x
        if self.diagnostic_normalizer is None:
            x, mean, std = normalize_batch(x, valid, normalize_type=f.normalize)
        else:
            x, mean, std = self.diagnostic_normalizer(x, valid)
        mask = torch.arange(x.size(-1), device=x.device).repeat(x.size(0), 1) >= valid.unsqueeze(1)
        features = x.masked_fill(mask.unsqueeze(1), f.pad_value)
        if self.include_stages:
            return features, valid, pairs, power, mel, log_mel, mean, std
        return features, valid
