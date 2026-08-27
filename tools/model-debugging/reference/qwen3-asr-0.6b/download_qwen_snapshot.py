"""Download the official Qwen3-ASR-0.6B snapshot without treating third-party ONNX as oracle."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def _disable_tls_verify() -> None:
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    import urllib3

    urllib3.disable_warnings()
    import requests

    original = requests.Session.request

    def unverified_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["verify"] = False
        return original(self, method, url, **kwargs)

    requests.Session.request = unverified_request  # type: ignore[method-assign]


def main() -> None:
    args = parse_args()
    _disable_tls_verify()

    from huggingface_hub import HfApi, snapshot_download
    try:
        from huggingface_hub.utils import get_session

        get_session().verify = False
    except Exception:  # noqa: BLE001
        pass

    api = HfApi()
    files = api.list_repo_files(args.repo_id, revision=args.revision)
    onnx_files = [name for name in files if name.lower().endswith((".onnx", ".onnx_data"))]
    inventory = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "file_count": len(files),
        "files": files,
        "onnx_files": onnx_files,
        "has_official_onnx": len(onnx_files) > 0,
    }
    try:
        info = api.model_info(args.repo_id, revision=args.revision)
        inventory["commit_sha"] = getattr(info, "sha", None)
    except Exception:  # noqa: BLE001
        inventory["commit_sha"] = None
    print(json.dumps(
        {k: inventory[k] for k in ("repo_id", "revision", "commit_sha", "file_count", "onnx_files", "has_official_onnx")},
        indent=2,
    ))
    if args.list_only:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        (args.local_dir / "_hf-file-list.json").write_text(
            json.dumps(inventory, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.local_dir),
    )
    print(f"Downloaded {args.repo_id}@{args.revision} to {args.local_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"download failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
