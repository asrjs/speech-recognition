from validate_variants import build_fixture_prompt_ids


class FakeTokenizer:
    eos_token_id = 50257

    ids = {
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|tr|>": 50268,
        "<|transcribe|>": 50359,
        "<|notimestamps|>": 50363,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids[token]


def test_prompt_ids_are_built_per_fixture_not_from_first_fixture():
    fixtures = [
        {"filename": "turkish.wav", "language": "tr"},
        {"filename": "jfk.wav", "language": "en"},
    ]

    prompts = build_fixture_prompt_ids(fixtures, FakeTokenizer())

    assert prompts["turkish.wav"]["prompt_ids"] == [50258, 50268, 50359, 50363]
    assert prompts["jfk.wav"]["prompt_ids"] == [50258, 50259, 50359, 50363]


def test_unknown_language_uses_default_english_prompt_consistently():
    fixtures = [
        {"filename": "hash.wav", "language": "unknown"},
        {"filename": "missing-language.wav"},
    ]

    prompts = build_fixture_prompt_ids(fixtures, FakeTokenizer())

    assert prompts["hash.wav"]["prompt_language"] == "en"
    assert prompts["missing-language.wav"]["prompt_language"] == "en"
    assert prompts["hash.wav"]["prompt_ids"] == prompts["missing-language.wav"]["prompt_ids"]


if __name__ == "__main__":
    test_prompt_ids_are_built_per_fixture_not_from_first_fixture()
    test_unknown_language_uses_default_english_prompt_consistently()
