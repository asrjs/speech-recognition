from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "canary-180m-flash"))
from canary_common import PromptSettings, build_prompt_slots, encode_prompt  # noqa: E402


class _Prompt:
    def __init__(self):
        self.dialog = None

    def encode_dialog(self, dialog):
        self.dialog = dialog
        return {"context_ids": [3, 4, 8, 4, 11]}


class _Model:
    def __init__(self, prompt_format: str):
        self.prompt_format = prompt_format
        self.cfg = SimpleNamespace(
            prompt_defaults=[
                SimpleNamespace(
                    slots={
                        "decodercontext": "",
                        "emotion": "<|emo:undefined|>",
                        "source_lang": "<|en|>",
                        "target_lang": "<|en|>",
                        "pnc": "<|pnc|>",
                        "itn": "<|noitn|>",
                        "timestamp": "<|notimestamp|>",
                        "diarize": "<|nodiarize|>",
                        "prompt_language": "spl_tokens",
                    }
                )
            ]
        )
        self.prompt = _Prompt()


class CanaryPromptTests(unittest.TestCase):
    def test_legacy_canary_uses_task_slot_and_exact_native_asr_prompt(self):
        model = _Model("canary")

        prompt_ids = encode_prompt(model, PromptSettings())

        self.assertEqual(prompt_ids, [3, 4, 8, 4, 11])
        self.assertEqual(
            model.prompt.dialog,
            [{"role": "user", "slots": {"source_lang": "en", "task": "asr", "target_lang": "en", "pnc": "yes"}}],
        )

    def test_legacy_canary_translation_uses_translate_task_and_nopnc(self):
        model = _Model("canary")

        slots = build_prompt_slots(
            model,
            PromptSettings(source_lang="en", target_lang="de", task="translation", pnc=False),
        )

        self.assertEqual(slots, {"source_lang": "en", "task": "translate", "target_lang": "de", "pnc": "no"})

    def test_canary2_slots_remain_unchanged(self):
        model = _Model("canary2")

        slots = build_prompt_slots(model, PromptSettings())

        self.assertEqual(slots["source_lang"], "<|en|>")
        self.assertEqual(slots["target_lang"], "<|en|>")
        self.assertEqual(slots["pnc"], "<|pnc|>")
        self.assertEqual(slots["itn"], "<|noitn|>")
        self.assertEqual(slots["timestamp"], "<|notimestamp|>")
        self.assertEqual(slots["diarize"], "<|nodiarize|>")
        self.assertEqual(slots["prompt_language"], "spl_tokens")
        self.assertNotIn("task", slots)


if __name__ == "__main__":
    unittest.main()
