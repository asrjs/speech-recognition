"""Actual CPU operator controls with synthetic PCM, not model acceptance."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from processor_export_stages import run_stft_input_control


class WindowFftControlTests(unittest.TestCase):
    def fixture(self, samples):
        torch.set_num_threads(4)
        f = SimpleNamespace(n_fft=512, hop_length=160, win_length=400,
            window=torch.hann_window(400, periodic=False), preemph=.97)
        waveform = torch.from_numpy(np.random.default_rng(17).normal(size=(1, samples)).astype(np.float32))
        signal = torch.cat((waveform[:, :1], waveform[:, 1:] - f.preemph * waveform[:, :-1]), dim=1).numpy()
        spectrum = torch.stft(torch.from_numpy(signal), n_fft=512, hop_length=160,
            win_length=400, window=f.window, center=True, pad_mode='constant',
            return_complex=True).numpy()
        return SimpleNamespace(featurizer=f), waveform, signal, spectrum

    def test_dynamic_frames_and_identical_input_fft_controls(self):
        for samples in (1600, 1773):
            with self.subTest(samples=samples):
                processor, waveform, signal, spectrum = self.fixture(samples)
                receipt = run_stft_input_control(processor, waveform,
                    torch.tensor([samples]), signal, spectrum)
                control = receipt['window_fft_control']
                self.assertTrue(control['native_frames_independent_bytes_exact'])
                self.assertTrue(control['native_fft_oracle_bytes_exact'])
                self.assertTrue(control['onnx_windowed_frames']['bytes_exact'])
                self.assertEqual(control['onnx_windowed_frames']['shape'], [1, samples // 160 + 1, 512])
                self.assertTrue(control['onnx_identical_frame_fft']['allclose'])

    def test_changed_oracle_fails_before_window_fft_certification(self):
        processor, waveform, signal, spectrum = self.fixture(1600)
        spectrum.real.flat[1] = np.nextafter(spectrum.real.flat[1], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, 'does not reproduce captured spectrum'):
            run_stft_input_control(processor, waveform,
                torch.tensor([1600]), signal, spectrum)


if __name__ == '__main__':
    unittest.main()
