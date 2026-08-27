"""Download official FunAudioLLM/SenseVoiceSmall weights. Not a third-party ONNX.

Uses curl --insecure on this machine because local SSL MITM breaks
huggingface_hub. Does not change git config.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEST = Path(r"N:\models\sensevoice\SenseVoiceSmall")
BASE = "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main"
FILES = [
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "config.yaml",
    "configuration.json",
    "demo.py",
    "requirements.txt",
    "README.md",
    "model.pt",
]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = DEST / name
        print(f"GET {name}")
        subprocess.check_call(
            ["curl.exe", "-k", "-L", "--retry", "5", "-C", "-", "-o", str(dest), f"{BASE}/{name}"]
        )
    src_model = Path(r"N:\github\FunAudioLLM\SenseVoice\model.py")
    if src_model.is_file():
        (DEST / "model.py").write_bytes(src_model.read_bytes())
    print(DEST)


if __name__ == "__main__":
    main()
