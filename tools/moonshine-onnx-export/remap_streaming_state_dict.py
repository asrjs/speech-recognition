"""Rewrite the streaming Moonshine state dict into the layout transformers expects.

The streaming checkpoint names the subsampler under an `embedder` submodule, uses
`linear` where the non-streaming model uses `conv3`, calls the groupnorm parameter
`comp.log_k`, and names every LayerNorm scale `gamma` instead of `weight`. The
module graph is otherwise identical, so a key rename is enough to load the real
weights instead of letting the loader silently initialize them.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

RENAMES = [
    ("model.encoder.embedder.conv1.", "model.encoder.conv1."),
    ("model.encoder.embedder.conv2.", "model.encoder.conv2."),
    ("model.encoder.embedder.linear.", "model.encoder.conv3."),
    ("model.encoder.embedder.comp.log_k", "model.encoder.groupnorm.log_k"),
    ("model.encoder.final_norm.gamma", "model.encoder.layer_norm.weight"),
]


def remap(name: str) -> str:
    for source, target in RENAMES:
        if name.startswith(source):
            return target + name[len(source):]
    if name.endswith("layernorm.gamma"):
        return name[: -len("gamma")] + "weight"
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/moonshine-streaming-tiny/model.safetensors"))
    parser.add_argument("--output", type=Path, default=Path(".cache/moonshine-streaming-tiny/model.remapped.safetensors"))
    args = parser.parse_args(argv)

    with args.source.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        payload = handle.read()

    base = 8 + header_len
    out_header: dict[str, object] = {}
    body = bytearray()
    renamed = 0
    for name, entry in header.items():
        if name == "__metadata__":
            out_header["__metadata__"] = entry
            continue
        start, end = entry["data_offsets"]
        raw = payload[start:end]
        new_name = remap(name)
        if new_name != name:
            renamed += 1
        record = dict(entry)
        record["data_offsets"] = [len(body), len(body) + len(raw)]
        out_header[new_name] = record
        body += raw

    encoded = json.dumps(out_header).encode("utf-8")
    padding = (8 - (len(encoded) + 8) % 8) % 8
    encoded += b" " * padding
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(bytes(body))

    print(json.dumps({
        "source": str(args.source),
        "output": str(args.output),
        "tensors": len(out_header) - 1,
        "renamed": renamed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
