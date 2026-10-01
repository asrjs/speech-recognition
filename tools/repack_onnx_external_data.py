"""Consolidate external ONNX tensor ranges without loading their weights.

Writes only to a fresh candidate directory. Copies each reference separately
(including valid source overlaps/aliases); removes stale external checksum and
basepath keys. Graph operations, tensor values and inline weights stay intact.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import onnx

from onnx_artifacts import external_ranges

BLOCK_BYTES = 8 * 1024 * 1024


def range_hash(path: Path, offset=0, length=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = path.stat().st_size - offset if length is None else length
        while remaining:
            block = stream.read(min(BLOCK_BYTES, remaining))
            if not block:
                raise ValueError(f"Unexpected EOF: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def repoint_tensor(tensor, location: str, offset: int, length: int):
    if offset < 0 or length <= 0:
        raise ValueError("Invalid destination range")
    tensor.ClearField("external_data")
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value


def copy_range(source: Path, offset: int, length: int, target) -> str:
    digest = hashlib.sha256()
    before = source.stat()
    with source.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(BLOCK_BYTES, remaining))
            if not block:
                raise ValueError(f"Unexpected EOF while copying: {source}")
            target.write(block)
            digest.update(block)
            remaining -= len(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while copying: {source}")
    return digest.hexdigest()


def repack(source: Path, output_dir: Path, expected_sha256: str, companions=()) -> dict:
    source = source.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"Graph missing/empty: {source}")
    source_hash = range_hash(source)
    if source_hash != expected_sha256:
        raise ValueError("Source graph SHA-256 mismatch; no candidate written")
    model = onnx.load(str(source), load_external_data=False)
    references = list(external_ranges(model, source))
    if not references:
        raise ValueError("Graph has no external tensors to repack")
    data_name = source.name + ".data"
    reserved = {source.name, data_name, "repack-receipt.json"}
    copies = []
    for name in companions:
        path = (source.parent / name).resolve()
        if not name or Path(name).name != name or path.parent != source.parent or name in reserved:
            raise ValueError(f"Companion name is unsafe or collides: {name!r}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Companion missing/empty: {path}")
        copies.append(path)
        reserved.add(name)
    # Run the source checker before creating any output. This uses the path,
    # not a >2 GiB in-memory model protobuf.
    onnx.checker.check_model(str(source))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "REPACK_IN_PROGRESS", "source_graph": {"path": str(source), "sha256": source_hash},
        "output_dir": str(output_dir), "block_bytes": BLOCK_BYTES,
        "policy": "copy_each_reference; remove external checksum/basepath; preserve inline tensors",
        "external_tensor_count": len(references), "source_external_file_count": len({r.path for r in references}),
        "ranges": [], "companions": [], "numerical_parity": "UNTESTED",
    }
    try:
        target_data = output_dir / data_name
        with target_data.open("xb") as stream:
            for reference in references:
                offset = stream.tell()
                digest = copy_range(reference.path, reference.offset, reference.length, stream)
                repoint_tensor(reference.tensor, data_name, offset, reference.length)
                receipt["ranges"].append({
                    "tensor": reference.tensor.name, "source_location": reference.location,
                    "source_offset": reference.offset, "length": reference.length,
                    "destination_offset": offset, "sha256": digest,
                })
        # Raw protobuf serialization intentionally avoids ONNX's external-data
        # writer/converter: the weight bytes are already safely streamed.
        target_graph = output_dir / source.name
        with target_graph.open("xb") as stream:
            stream.write(model.SerializeToString())
        for path in copies:
            target = output_dir / path.name
            with target.open("xb") as stream:
                digest = copy_range(path, 0, path.stat().st_size, stream)
            if range_hash(target) != digest:
                raise ValueError(f"Companion copy hash mismatch: {path.name}")
            receipt["companions"].append({"name": path.name, "bytes": target.stat().st_size, "sha256": digest})
        # Independent post-write reads prove exact tensor-range byte identity.
        for reference, item in zip(references, receipt["ranges"]):
            if (range_hash(reference.path, reference.offset, reference.length) != item["sha256"]
                    or range_hash(target_data, item["destination_offset"], item["length"]) != item["sha256"]):
                raise ValueError(f"Tensor range copy hash mismatch: {reference.tensor.name}")
        if range_hash(source) != source_hash:
            raise ValueError("Source graph changed during repacking")
        packed = onnx.load(str(target_graph), load_external_data=False)
        packed_ranges = list(external_ranges(packed, target_graph))
        if len(packed_ranges) != len(references) or {r.location for r in packed_ranges} != {data_name}:
            raise ValueError("Packed external-data contract changed unexpectedly")
        onnx.checker.check_model(str(target_graph))
        receipt["graph"] = {"name": target_graph.name, "bytes": target_graph.stat().st_size,
                            "sha256": range_hash(target_graph)}
        receipt["external_data"] = {"name": data_name, "bytes": target_data.stat().st_size,
                                   "sha256": range_hash(target_data)}
        receipt["status"] = "REPACKED_BYTE_IDENTICAL_UNVALIDATED"
        return receipt
    except Exception as error:
        receipt["status"] = "REPACK_FAILED"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        with (output_dir / "repack-receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-file", action="append", default=[])
    args = parser.parse_args()
    receipt = repack(args.graph, args.output_dir, args.expected_sha256, args.copy_file)
    print(json.dumps({key: receipt[key] for key in ("status", "external_tensor_count", "graph", "external_data")}))


if __name__ == "__main__":
    main()
