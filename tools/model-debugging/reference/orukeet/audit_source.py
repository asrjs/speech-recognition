"""Audit the original .nemo configuration and every token against local v3 assets.

This checks configuration/tokenizer reuse only; numerical and browser parity
remain independent acceptance gates.
"""

from __future__ import annotations

import argparse
import datetime
import json
import tarfile
from pathlib import Path

import sentencepiece as spm
import yaml

from acquire_source import SOURCE_SHA256, sha256
from artifact_outputs import require_new_outputs, write_new_text


def indexed_vocab(path: Path) -> list[str]:
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        piece, token_id = line.rsplit(" ", 1)
        token_id = int(token_id)
        if token_id in by_id:
            raise ValueError(f"Duplicate token ID in {path}: {token_id}")
        by_id[token_id] = piece
    if sorted(by_id) != list(range(len(by_id))):
        raise ValueError(f"Non-contiguous vocabulary IDs in {path}")
    return [by_id[index] for index in range(len(by_id))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", type=Path, default=Path("models/orukeet/source/orukeet-v0.1.0.nemo"))
    parser.add_argument("--v3-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_new_outputs([args.output], protected_inputs=[args.nemo, args.v3_vocab])
    source_hash = sha256(args.nemo)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError("Original .nemo SHA-256 mismatch; stop before auditing configuration")
    with tarfile.open(args.nemo, "r:*") as archive:
        cfg = yaml.safe_load(archive.extractfile("./model_config.yaml").read())
        model_member = str(cfg["tokenizer"]["model_path"]).removeprefix("nemo:")
        processor = spm.SentencePieceProcessor(model_proto=archive.extractfile(f"./{model_member}").read())
    original = [processor.id_to_piece(index) for index in range(processor.vocab_size())]
    original.append("<blk>")
    baseline = indexed_vocab(args.v3_vocab)
    mismatches = [
        {"id": index, "original": left, "v3": right}
        for index, (left, right) in enumerate(zip(original, baseline)) if left != right
    ]
    tokenizer_match = len(original) == len(baseline) and not mismatches
    actual = {
        "model_class": cfg["target"], "loss_name": cfg["loss"]["loss_name"],
        "sample_rate": cfg["preprocessor"]["sample_rate"],
        "mel_bins": cfg["preprocessor"]["features"],
        "window_size": cfg["preprocessor"]["window_size"],
        "window_stride": cfg["preprocessor"]["window_stride"],
        "normalize": cfg["preprocessor"]["normalize"], "n_fft": cfg["preprocessor"]["n_fft"],
        "subsampling_factor": cfg["encoder"]["subsampling_factor"],
        "encoder_hidden": cfg["encoder"]["d_model"],
        "prediction_hidden": cfg["decoder"]["prednet"]["pred_hidden"],
        "prediction_layers": cfg["decoder"]["prednet"]["pred_rnn_layers"],
        "vocabulary_size_with_blank": len(original), "blank_id": processor.vocab_size(),
        "duration_values": cfg["model_defaults"]["tdt_durations"],
        "duration_channels": cfg["joint"]["num_extra_outputs"],
    }
    expected = {
        "model_class": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "loss_name": "tdt", "sample_rate": 16000, "mel_bins": 128,
        "window_size": 0.025, "window_stride": 0.01, "normalize": "per_feature", "n_fft": 512,
        "subsampling_factor": 8, "encoder_hidden": 1024,
        "prediction_hidden": 640, "prediction_layers": 2,
        "vocabulary_size_with_blank": 8193, "blank_id": 8192,
        "duration_values": [0, 1, 2, 3, 4], "duration_channels": 5,
    }
    checks = {name: {"actual": value, "expected": expected[name], "pass": value == expected[name]}
              for name, value in actual.items()}
    config_match = all(check["pass"] for check in checks.values())
    record = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "COMPATIBLE_CONFIG_TOKENIZER" if config_match and tokenizer_match else "MISMATCH",
        "original_weight_sha256": source_hash,
        "v3_vocab": {"path": str(args.v3_vocab.resolve()), "sha256": sha256(args.v3_vocab)},
        "config_checks": checks,
        "tokenizer": {"pass": tokenizer_match, "original_count": len(original),
                      "v3_count": len(baseline), "mismatch_count": len(mismatches),
                      "first_mismatches": mismatches[:10]},
        "runtime_config": {
            "sampleRate": actual["sample_rate"], "melBins": actual["mel_bins"],
            "frameShiftSeconds": actual["window_stride"], "subsamplingFactor": actual["subsampling_factor"],
            "predictionHiddenSize": actual["prediction_hidden"], "predictionLayers": actual["prediction_layers"],
            "vocabularySize": len(original), "tokenizer": {"kind": "sentencepiece", "blankTokenId": actual["blank_id"]},
        },
        "numerical_parity": "UNTESTED", "browser_parity": "UNTESTED",
    }
    write_new_text(args.output, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"{record['status']}: {len(original)} token IDs; {len(mismatches)} differences; {args.output}")
    if not config_match or not tokenizer_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
