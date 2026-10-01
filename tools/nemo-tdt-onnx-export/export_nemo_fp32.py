"""Export official offline NeMo TDT weights to this family's two fp32 graphs.

NeMo owns encoder and combined predictor/joint export preparation, including
no-extra-SOS prediction and raw joint logits. This is not a search/oracle
implementation. Numerical, library and browser parity remain separate gates.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_artifacts import external_ranges, iter_tensors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_output_directory(path: Path) -> Path:
    """Never overwrite a prior graph, receipt or partially failed export."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_tdt_model(model) -> None:
    if model.cfg.loss.loss_name != "tdt":
        raise ValueError("Only NeMo TDT models are supported by this exporter")
    if list(model.list_export_subnets()) != ["encoder", "decoder_joint"]:
        raise ValueError("Expected offline encoder + decoder_joint export subnets")
    if getattr(model.encoder, "export_cache_support", False) or hasattr(model, "prompt_kernel"):
        raise ValueError("Cache-aware/prompted families need their own export contract")
    if not model.decoder.blank_as_pad:
        raise ValueError("Combined TDT runtime requires a zero-embedding blank/SOS")


def audit_graph(path: Path, expected_inputs: dict, expected_outputs: dict) -> dict:
    """Check actual names, dtypes/ranks and every colocated external-data range."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    contract = {}
    for kind, values, expected in (
        ("inputs", graph.graph.input, expected_inputs),
        ("outputs", graph.graph.output, expected_outputs),
    ):
        actual = {}
        for value in values:
            tensor = value.type.tensor_type
            dims = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor.shape.dim]
            actual[value.name] = {"dtype": onnx.TensorProto.DataType.Name(tensor.elem_type), "shape": dims}
        if set(actual) != set(expected):
            raise ValueError(f"{path.name} {kind} names differ: {list(actual)}")
        for name, (dtype, rank) in expected.items():
            if actual[name]["dtype"] != dtype or len(actual[name]["shape"]) != rank:
                raise ValueError(f"{path.name} {name} dtype/rank differs: {actual[name]}")
        contract[kind] = actual
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in iter_tensors(graph)):
        raise ValueError(f"{path.name} contains fp16 tensors in an fp32 export")
    external = {}
    references = list(external_ranges(graph, path))
    for reference in references:
        external[reference.location] = reference.path
    # Path-based checking handles models whose weights exceed protobuf's 2 GB limit.
    onnx.checker.check_model(str(path))
    return {
        **contract,
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "opsets": [{"domain": item.domain, "version": item.version} for item in graph.opset_import],
        "node_count": len(graph.graph.node),
        "external_tensor_count": len(references),
        "external_data": [{"location": name, "bytes": item.stat().st_size, "sha256": sha256(item)}
                          for name, item in sorted(external.items())],
    }


ENCODER_INPUTS = {"audio_signal": ("FLOAT", 3), "length": ("INT64", 1)}
ENCODER_OUTPUTS = {"outputs": ("FLOAT", 3), "encoded_lengths": ("INT64", 1)}
DECODER_INPUTS = {
    "encoder_outputs": ("FLOAT", 3), "targets": ("INT32", 2), "target_length": ("INT32", 1),
    "input_states_1": ("FLOAT", 3), "input_states_2": ("FLOAT", 3),
}
DECODER_OUTPUTS = {
    "outputs": ("FLOAT", 4), "prednet_lengths": ("INT32", 1),
    "output_states_1": ("FLOAT", 3), "output_states_2": ("FLOAT", 3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--trace-frames", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/nemo-reference"))
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.trace_frames < 16 or args.opset < 17:
        raise ValueError("Positive CPU threads, >=16 trace frames and opset >=17 required")
    source = args.nemo.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Original weight file missing/empty: {source}")
    actual_hash = sha256(source)
    if actual_hash != args.expected_sha256:
        raise ValueError("Original weight SHA-256 mismatch; no export written")
    output = create_output_directory(args.output_dir)
    receipt = {
        "schema_version": 1, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "EXPORT_IN_PROGRESS", "model_id": args.model_id, "model_revision": args.model_revision,
        "source": {"path": str(source), "sha256": actual_hash, "bytes": source.stat().st_size},
        "settings": {"dtype": "fp32", "device": "cpu", "cpu_threads": args.cpu_threads,
                     "trace_frames": args.trace_frames, "opset": args.opset, "seed": 0,
                     "use_dynamo": False, "quantization": None},
        "numerical_parity": "UNTESTED", "library_parity": "UNTESTED", "browser_parity": "UNTESTED",
    }
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        cache = args.cache_dir.resolve() / "numba"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache)
        import torch
        from nemo.collections.asr.models import ASRModel
        import numba
        from omegaconf import OmegaConf

        numba.config.reload_config()
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(0)
        receipt["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                  "torch": torch.__version__, "nemo_toolkit": version("nemo_toolkit"),
                                  "onnx": version("onnx"), "numba_cache": str(cache)}
        print(f"Restoring official weights: {source}", flush=True)
        model = ASRModel.restore_from(str(source), map_location="cpu").eval().float()
        require_tdt_model(model)
        cfg = OmegaConf.to_container(model.cfg, resolve=True)
        (output / "source-config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        blank_id = int(model.decoder.blank_idx)
        pieces = model.tokenizer.ids_to_tokens(list(range(blank_id)))
        if len(pieces) != blank_id or any("\n" in piece or "\r" in piece for piece in pieces):
            raise ValueError("Invalid indexed tokenizer vocabulary")
        (output / "vocab.txt").write_text("".join(f"{piece} {index}\n" for index, piece in enumerate([*pieces, "<blk>"])), encoding="utf-8")
        export_options = {"onnx_opset_version": args.opset, "use_dynamo": False, "check_trace": False}
        encoder_path = output / "encoder-model.onnx"
        print("Exporting fp32 encoder with NeMo's official preparation ...", flush=True)
        model.encoder.export(str(encoder_path), input_example=model.encoder.input_example(max_dim=args.trace_frames), **export_options)
        receipt["encoder"] = audit_graph(encoder_path, ENCODER_INPUTS, ENCODER_OUTPUTS)
        # Use a single frame/label trace, not RNNTJoint's 8192-frame default.
        # Dimensions come from the restored family, never a branded model name.
        labels, _, states = model.decoder.input_example(max_dim=1)
        decoder_inputs = (torch.randn(1, model.joint.encoder_hidden, 1), labels,
                          torch.ones(1, dtype=torch.int32), *states)
        decoder_path = output / "decoder_joint-model.onnx"
        print("Exporting fp32 combined predictor/joint with NeMo's official preparation ...", flush=True)
        model.decoder_joint.export(str(decoder_path), input_example=decoder_inputs, **export_options)
        receipt["decoder_joint"] = audit_graph(decoder_path, DECODER_INPUTS, DECODER_OUTPUTS)
        receipt["vocab"] = {"count_with_blank": blank_id + 1, "blank_id": blank_id,
                             "sha256": sha256(output / "vocab.txt")}
        receipt["status"] = "EXPORTED_UNVALIDATED"
        print(f"Exported two fp32 graphs; numerical/library/browser parity UNTESTED: {output}", flush=True)
    except Exception as error:
        receipt["status"] = "EXPORT_FAILED"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        (output / "export-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
