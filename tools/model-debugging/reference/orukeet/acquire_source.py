"""Acquire pinned original Orukeet weights and provenance for the porting oracle.

Uses the existing huggingface_hub downloader for resumable large files. Model
weights and cached metadata stay in ignored models/, never in source control.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from artifact_outputs import require_new_outputs


REPO_ID = "oruk/orukeet"
REVISION = "b59c13a733fce5cf193230d0fa317ee7f145a108"
SOURCE_FILE = "orukeet-v0.1.0.nemo"
SOURCE_BYTES = 2509342720
SOURCE_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"
SMALL_FILES = (
    "config.json", "processor_config.json", "tokenizer.json",
    "LICENSE-WEIGHTS", "NOTICE.md", "ARTIFACTS.json", "README.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("models/orukeet/source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    require_new_outputs([args.output], protected_directories=[args.source_dir])
    record = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "status": "IN_PROGRESS",
        "source_dir": str(args.source_dir.resolve()),
        "files": [],
        "parity_status": "UNTESTED",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_file = args.output.open("x", encoding="utf-8")

    def save() -> None:
        # Update only the new receipt handle owned by this acquisition run.
        receipt_file.seek(0)
        receipt_file.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        receipt_file.truncate()
        receipt_file.flush()

    try:
        save()
        info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise RuntimeError(f"Source revision mismatch: {info.sha}")
        source = next(item for item in info.siblings if item.rfilename == SOURCE_FILE)
        if source.size != SOURCE_BYTES or source.lfs.sha256 != SOURCE_SHA256:
            raise RuntimeError("Pinned original-weight metadata does not match expected bytes/hash")
        record["license"] = info.card_data.license
        record["languages"] = info.card_data.language
        record["original_weight"] = {
            "filename": SOURCE_FILE, "bytes": SOURCE_BYTES,
            "expected_sha256": SOURCE_SHA256, "downloaded": False,
        }
        for name in (*SMALL_FILES, *((SOURCE_FILE,) if args.download_weights else ())):
            print(f"Acquire {REPO_ID}@{REVISION[:12]} / {name}", flush=True)
            downloaded = Path(hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=args.source_dir))
            digest = sha256(downloaded)
            item = {"filename": name, "bytes": downloaded.stat().st_size, "sha256": digest}
            if name == SOURCE_FILE:
                if digest != SOURCE_SHA256 or item["bytes"] != SOURCE_BYTES:
                    raise RuntimeError("Downloaded original-weight size/SHA-256 mismatch")
                record["original_weight"].update(downloaded=True, sha256=digest, verified=True)
            record["files"].append(item)
            save()
        config = json.loads((args.source_dir / "config.json").read_text(encoding="utf-8"))
        record["converted_config_audit"] = {
            "architecture": config["architectures"], "vocab_size": config["vocab_size"],
            "blank_token_id": config["blank_token_id"], "durations": config["durations"],
            "mel_bins": config["encoder_config"]["num_mel_bins"],
            "subsampling_factor": config["encoder_config"]["subsampling_factor"],
            "prediction_hidden_size": config["decoder_hidden_size"],
            "prediction_layers": config["num_decoder_layers"],
            "note": "Converted metadata only; inspect original .nemo config before declaring compatibility.",
        }
        record["status"] = "SOURCE_VERIFIED" if args.download_weights else "METADATA_ONLY"
        save()
        print(f"{record['status']}: {args.output}", flush=True)
    except Exception as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        receipt_file.close()


if __name__ == "__main__":
    main()
