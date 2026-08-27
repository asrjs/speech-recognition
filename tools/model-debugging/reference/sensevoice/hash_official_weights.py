"""Record hashes for the local official SenseVoiceSmall snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEST = Path(r"N:\models\sensevoice\SenseVoiceSmall")
OUT = Path(r"N:\github\asrjs\speech-recognition\tools\data\results\sensevoice\sensevoice-small-provenance.json")

SKIP_SUFFIXES = {".png", ".jpg", ".mp3"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(item for item in DEST.iterdir() if item.is_file()):
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append({
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = {
        "schema_version": 1,
        "hf_repo": "FunAudioLLM/SenseVoiceSmall",
        "hf_revision": "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
        "git_clone": r"N:\github\FunAudioLLM\SenseVoice",
        "git_revision": "6991744856587fa44379e8b5dcc432debffeb1be",
        "license": "FunASR Model Open Source License (model-license)",
        "model_dir": str(DEST),
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
