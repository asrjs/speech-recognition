"""Model-neutral ONNX tensor traversal and colocated external-data validation.

Tooling only: no speech decoding, NeMo, Torch, or published runtime dependency.
"""

from dataclasses import dataclass
from pathlib import Path

import onnx


def iter_tensors(message):
    """Visit tensors in initializers, attributes, subgraphs, sparse data/functions.

    Protobuf reflection covers all actual TensorProto fields without relying on
    ONNX's private initializer-only or attribute traversal helpers.
    """
    if message.DESCRIPTOR.full_name == onnx.TensorProto.DESCRIPTOR.full_name:
        yield message
        return
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        children = value if field.label == field.LABEL_REPEATED else (value,)
        for child in children:
            yield from iter_tensors(child)


@dataclass(frozen=True)
class ExternalRange:
    tensor: object
    location: str
    path: Path
    offset: int
    length: int


def external_ranges(model, graph_path: Path):
    """Validate every external reference before any consumer copies its bytes."""
    base = graph_path.resolve().parent
    for tensor in iter_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {entry.key: entry.value for entry in tensor.external_data}
        if len(info) != len(tensor.external_data):
            raise ValueError(f"Duplicate external-data key: {tensor.name}")
        if tensor.HasField("raw_data"):
            raise ValueError(f"External tensor also has embedded raw_data: {tensor.name}")
        location = info.get("location", "")
        target = (base / location).resolve()
        if not location or Path(location).is_absolute() or target.parent != base:
            raise ValueError(f"External tensor {tensor.name} is not colocated: {location!r}")
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"External tensor file missing/empty: {target}")
        size = target.stat().st_size
        offset = int(info.get("offset", "0"))
        length = int(info.get("length", str(size - offset)))
        if offset < 0 or length <= 0 or offset + length > size:
            raise ValueError(f"External tensor range invalid: {tensor.name}")
        yield ExternalRange(tensor, location, target, offset, length)
