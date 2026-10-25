from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "canary-180m-flash"))
import canary_common
import verify_canary_onnx


V2_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
]
V2_CHECKPOINT = Path(os.environ.get("CANARY_V2_CHECKPOINT", ""))


class LocalCheckpointLoadingTests(unittest.TestCase):
    def _model_mock(self) -> Mock:
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model
        return model

    def test_existing_nemo_path_uses_restore_from(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nemo") as checkpoint:
            model = self._model_mock()
            with (
                patch.object(canary_common.ASRModel, "restore_from", return_value=model) as restore,
                patch.object(canary_common.ASRModel, "from_pretrained") as from_pretrained,
            ):
                result = canary_common.load_model(checkpoint.name, "cpu")

        self.assertIs(result, model)
        restore.assert_called_once_with(str(Path(checkpoint.name).resolve()), map_location="cpu")
        from_pretrained.assert_not_called()


    def test_missing_nemo_path_does_not_fall_back_to_hub(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.nemo"
            with patch.object(canary_common.ASRModel, "from_pretrained") as from_pretrained:
                with self.assertRaisesRegex(FileNotFoundError, "Local NeMo checkpoint"):
                    canary_common.load_model(str(missing), "cpu")
            from_pretrained.assert_not_called()

    def test_hub_identifier_keeps_from_pretrained_path(self) -> None:
        model = self._model_mock()
        with (
            patch.object(canary_common.ASRModel, "from_pretrained", return_value=model) as from_pretrained,
            patch.object(canary_common.ASRModel, "restore_from") as restore,
        ):
            result = canary_common.load_model("nvidia/canary-180m-flash", "cpu")

        self.assertIs(result, model)
        from_pretrained.assert_called_once_with("nvidia/canary-180m-flash", map_location="cpu")
        restore.assert_not_called()

    def test_explicit_checkpoint_keeps_hub_id_out_of_from_pretrained(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nemo") as checkpoint:
            model = self._model_mock()
            with (
                patch.object(canary_common.ASRModel, "restore_from", return_value=model) as restore,
                patch.object(canary_common.ASRModel, "from_pretrained") as from_pretrained,
            ):
                result = canary_common.load_model(
                    "nvidia/canary-1b-v2",
                    "cpu",
                    Path(checkpoint.name),
                )

        self.assertIs(result, model)
        restore.assert_called_once_with(str(Path(checkpoint.name).resolve()), map_location="cpu")
        from_pretrained.assert_not_called()


class CanaryGraphPublicationTests(unittest.TestCase):
    def test_external_graph_is_published_with_one_checked_sibling_sidecar(self) -> None:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        from export_canary_onnx import publish_graph

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "bundle"
            staging_dir = root / "staging"
            raw_dir.mkdir()
            output_dir.mkdir()
            staging_dir.mkdir()
            raw_path = raw_dir / "encoder-model.onnx"
            output_path = output_dir / "encoder-model.onnx"
            expected = np.arange(128, dtype=np.float32)
            graph = helper.make_graph(
                [],
                "canary-test",
                [],
                [],
                initializer=[numpy_helper.from_array(expected, name="weights")],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            onnx.save_model(
                model,
                str(raw_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="source-weights.data",
                size_threshold=0,
            )

            external_data = publish_graph(raw_path, output_path, staging_dir)

            self.assertEqual(external_data, "encoder-model.onnx.data")
            self.assertTrue((output_dir / external_data).is_file())
            published = onnx.load(str(output_path), load_external_data=True)
            np.testing.assert_array_equal(
                numpy_helper.to_array(published.graph.initializer[0]),
                expected,
            )


class CanaryVerifierGateTests(unittest.TestCase):
    def _reference_feature_result(self) -> dict:
        return {
            "preprocessor": {"skipped": True},
            "encoder": {
                "shape_match": True,
                "max_abs_diff": 1.0e9,
                "lengths_match": True,
                "mask": {"shape_match": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0},
            },
            "decode": {"token_ids_match": True, "text_match": True},
        }

    def test_skipped_frontend_does_not_claim_frontend_pass_and_numeric_diff_is_ungated(self) -> None:
        result = verify_canary_onnx.evaluate_verification(self._reference_feature_result())

        self.assertEqual(result["status"], "PASS_GATED_COMPONENTS")
        self.assertEqual(result["scope"]["preprocessor"], "SKIPPED_REFERENCE_FEATURES")
        self.assertEqual(result["scope"]["browser_runtime"], "NOT_RUN")
        self.assertEqual(result["gates"]["preprocessor_shape_and_lengths"], "SKIPPED")
        self.assertEqual(result["gates"]["numeric_differences"], "MEASURED_NOT_GATED")

    def test_decoder_mismatch_emits_failure_and_nonzero_exit(self) -> None:
        import argparse

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "verification.json"
            args = argparse.Namespace(output=output)
            result = self._reference_feature_result()
            result["decode"]["token_ids_match"] = False
            with (
                patch.object(verify_canary_onnx, "parse_args", return_value=args),
                patch.object(verify_canary_onnx, "run_verification", return_value=result),
            ):
                exit_code = verify_canary_onnx.main()

            recorded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(recorded["status"], "FAIL_COMPONENT_GATES")
            self.assertIn("decoder_token_ids_mismatch", recorded["failures"])

    def test_mask_value_mismatch_fails_exact_binary_mask_gate(self) -> None:
        result = self._reference_feature_result()
        result["encoder"]["mask"]["max_abs_diff"] = 1.0

        evaluated = verify_canary_onnx.evaluate_verification(result)

        self.assertEqual(evaluated["status"], "FAIL_COMPONENT_GATES")
        self.assertIn("encoder_mask_value_mismatch", evaluated["failures"])


@unittest.skipUnless(V2_CHECKPOINT.is_file(), "set CANARY_V2_CHECKPOINT to run real tokenizer metadata checks")
class CanaryV2UnifiedTokenizerTests(unittest.TestCase):
    def test_embedded_unified_vocab_matches_sentencepiece_and_export_payload(self) -> None:
        import sentencepiece as sentencepiece
        import sentencepiece.sentencepiece_model_pb2 as sentencepiece_model_pb2
        import yaml

        self.assertIsNotNone(canary_common.CanaryBPETokenizer)
        with tarfile.open(V2_CHECKPOINT, "r:*") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers()}
            config_bytes = archive.extractfile(members["model_config.yaml"]).read()
            tokenizer_member = members["cc5d48e83aad4be48aa9fa264b727c4b_tokenizer.model"]
            tokenizer_bytes = archive.extractfile(tokenizer_member).read()

        model_config = yaml.safe_load(config_bytes)
        self.assertEqual(model_config["prompt_format"], "canary2")
        self.assertEqual(model_config["preprocessor"]["features"], 128)
        self.assertEqual(model_config["encoder"]["n_layers"], 32)
        self.assertEqual(model_config["transf_decoder"]["config_dict"]["num_layers"], 8)
        self.assertEqual(model_config["head"]["num_classes"], 16384)

        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(tokenizer_bytes)
        self.assertFalse(proto.trainer_spec.byte_fallback)
        self.assertEqual(len(proto.pieces), 16384)

        native = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_bytes)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_path = Path(directory) / "tokenizer.model"
            tokenizer_path.write_bytes(tokenizer_bytes)
            tokenizer = canary_common.CanaryBPETokenizer(str(tokenizer_path))

            native_pieces = [native.id_to_piece(token_id) for token_id in range(native.get_piece_size())]
            self.assertEqual(tokenizer.vocab_size, 16384)
            self.assertEqual(list(tokenizer.vocab), native_pieces)
            self.assertEqual(tokenizer.ids_to_tokens(list(range(16384))), native_pieces)
            self.assertEqual(
                [tokenizer.token_to_id(piece) for piece in native_pieces],
                list(range(16384)),
            )

            model = SimpleNamespace(tokenizer=tokenizer, prompt_format="canary2")
            payload = canary_common.exported_tokenizer_payload(model, V2_LANGUAGES)
            unified = payload["subtokenizers"]["spl_tokens"]
            self.assertEqual(payload["language_codes"], V2_LANGUAGES)
            self.assertEqual(payload["vocab_size"], 16384)
            self.assertEqual(unified["offset"], 0)
            self.assertEqual(unified["size"], 16384)
            self.assertEqual(unified["pieces"], native_pieces)

            native_token_to_id = tokenizer.token_to_id

            def wrong_prompt_token_id(piece: str) -> int:
                if piece == "<|startofcontext|>":
                    return 0
                return native_token_to_id(piece)

            with patch.object(tokenizer, "token_to_id", side_effect=wrong_prompt_token_id):
                with self.assertRaisesRegex(ValueError, "token_to_id mismatch"):
                    canary_common.exported_tokenizer_payload(model, V2_LANGUAGES)

            expected_specials = {
                piece: token_id
                for token_id, piece in enumerate(native_pieces)
                if piece == "<pad>" or (piece.startswith("<|") and piece.endswith("|>"))
            }
            self.assertEqual(payload["special_tokens"], expected_specials)
            self.assertEqual(
                canary_common.special_token_id_set(tokenizer),
                set(expected_specials.values()),
            )
            for token in (
                "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
                "<|en|>", "<|pnc|>", "<|nopnc|>", "<|itn|>", "<|noitn|>",
                "<|timestamp|>", "<|notimestamp|>", "<|diarize|>", "<|nodiarize|>",
                "<|endoftext|>", "<|nospeech|>", "<pad>",
            ):
                self.assertEqual(payload["special_tokens"][token], native.piece_to_id(token))

            self.assertEqual(tokenizer.bos_id, native.piece_to_id("<|startoftranscript|>"))
            self.assertEqual(tokenizer.eos_id, native.piece_to_id("<|endoftext|>"))
            self.assertEqual(tokenizer.pad_id, native.piece_to_id("<pad>"))

            config_model = SimpleNamespace(
                tokenizer=tokenizer,
                prompt_format="canary2",
                cfg=SimpleNamespace(
                    preprocessor=SimpleNamespace(
                        normalize="per_feature",
                        sample_rate=16000,
                        features=128,
                        window_stride=0.01,
                    ),
                    encoder=SimpleNamespace(subsampling_factor=8),
                    model_defaults=SimpleNamespace(asr_enc_hidden=1024, lm_dec_hidden=1024),
                    transf_decoder=SimpleNamespace(config_dict=SimpleNamespace(max_sequence_length=1024)),
                    prompt_format="canary2",
                ),
            )
            runtime_config = canary_common.runtime_config_payload(
                "nvidia/canary-1b-v2",
                config_model,
                V2_LANGUAGES,
                {"encoder": "encoder-model.onnx.data", "decoder": None},
            )
            self.assertEqual(runtime_config["preprocessor_valid_length_mode"], "onnx")
            self.assertEqual(runtime_config["preprocessor_normalization"], "per_feature")
            self.assertEqual(
                runtime_config["external_data"],
                {"encoder": "encoder-model.onnx.data", "decoder": None},
            )

            text_cases = {}
            for label, text in (("english", "This is a short test."), ("punctuation", "Hello, world!")):
                ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(text))
                ids = [int(token_id) for token_id in ids]
                native_text = tokenizer.ids_to_text(ids)
                helper_text = canary_common.decode_text(tokenizer, ids)
                naive_piece_join = "".join(tokenizer.ids_to_tokens(ids)).replace("\u2581", " ").strip()
                self.assertEqual(helper_text, native_text.replace("\u2581", " ").strip())
                text_cases[label] = {
                    "ids": ids,
                    "nativeSentencePieceText": native_text,
                    "helperText": helper_text,
                    "naiveJoinedPieces": naive_piece_join,
                    "naiveMatchesNative": naive_piece_join == native_text,
                }

            special_and_text_ids = [tokenizer.token_to_id("<|en|>"), *tokenizer.tokens_to_ids(tokenizer.text_to_tokens("A test.")), tokenizer.eos_id]
            text_cases["specialFiltered"] = {
                "ids": special_and_text_ids,
                "nativeAfterSpecialFilter": tokenizer.ids_to_text(
                    [token_id for token_id in special_and_text_ids if token_id not in expected_specials.values()]
                ),
                "helperText": canary_common.decode_text(tokenizer, special_and_text_ids),
            }
            text_cases["unknownId"] = {
                "id": 0,
                "piece": tokenizer.vocab[0],
                "nativeSentencePieceText": tokenizer.ids_to_text([0]),
                "helperText": canary_common.decode_text(tokenizer, [0]),
            }
            print("CANARY_V2_TOKENIZER_CASES=" + json.dumps(text_cases, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
