"""Focused tests for the split-graph reference runner."""

from types import SimpleNamespace
import unittest

import numpy as np

from generate_hf_reference import build_decoder_step_feed


class DecoderStepFeedTests(unittest.TestCase):
    def test_cache_position_tracks_decoder_cache_length(self):
        session = SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="cache_position"),
            ]
        )
        step_input = np.array([[123]], dtype=np.int64)

        for cache_length, expected in ((4, [4]), (5, [5]), (8, [8])):
            past_kv = {
                "past_key_values.0.decoder.key": np.zeros((1, 20, cache_length, 64), dtype=np.float16),
                "past_key_values.0.decoder.value": np.zeros((1, 20, cache_length, 64), dtype=np.float16),
            }
            feed = build_decoder_step_feed(session, step_input, past_kv)
            np.testing.assert_array_equal(feed["cache_position"], np.array(expected, dtype=np.int64))
            self.assertEqual(feed["cache_position"].dtype, np.int64)
            np.testing.assert_array_equal(feed["input_ids"], step_input)

        multi_token_input = np.array([[123, 124, 125]], dtype=np.int64)
        past_kv = {
            "past_key_values.0.decoder.key": np.zeros((1, 20, 8, 64), dtype=np.float16),
        }
        feed = build_decoder_step_feed(session, multi_token_input, past_kv)
        np.testing.assert_array_equal(feed["cache_position"], np.array([8, 9, 10], dtype=np.int64))

    def test_legacy_decoder_step_without_cache_position_is_unchanged(self):
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_ids")])
        step_input = np.array([[123]], dtype=np.int64)
        past_kv = {"past_key_values.0.decoder.key": np.zeros((1, 20, 4, 64), dtype=np.float16)}

        feed = build_decoder_step_feed(session, step_input, past_kv)

        self.assertNotIn("cache_position", feed)
        np.testing.assert_array_equal(feed["input_ids"], step_input)

    def test_missing_decoder_cache_is_rejected_when_position_is_required(self):
        session = SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="cache_position"),
            ]
        )

        with self.assertRaisesRegex(ValueError, "no decoder self-attention cache"):
            build_decoder_step_feed(session, np.array([[123]], dtype=np.int64), {})


if __name__ == "__main__":
    unittest.main()
