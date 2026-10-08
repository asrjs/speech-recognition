"""Redux packing parity: kestrel dense-oracle materialization vs independent dequant.

The reference is this repo's independently validated tools/parakeet-ternary/ternary.py
(base-3 archive unpack + manifest weight rule). The official side is
kestrel_kernels' TernaryWeight.dequantized — the mode CUDA uses and the only
resident form available on this workstation: the native auto probe reports
'scalar', and set_gemm_isa('avx2') is accepted but the gemm itself raises
"no 'avx2' matrix-multiply path on this machine" (this Skylake-W Xeon lacks
VNNI), so packed int8 compute cannot run here at all. resident_form is therefore
overridden to the official dense/oracle branch for CPU in this probe only.

Expected result: bit-exact weights (both sides compute float32 scale * (code-1))
and bit-exact forwards. This validates the packed-format understanding both ways;
it is not the native kernel, not model inference and not a speed claim.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools' / 'parakeet-ternary'))
import ternary  # noqa: E402  (repo-local independent dequantizer)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import kestrel_kernels.ternary as kernels_ternary  # noqa: E402

_original_resident_form = kernels_ternary.resident_form


def _dense_oracle_resident_form(device):
    if torch.device(device).type == 'cpu':
        return 'dense'  # official oracle form; this CPU has no int8 gemm path
    return _original_resident_form(device)


kernels_ternary.resident_form = _dense_oracle_resident_form
from kestrel.models.parakeet_tdt.weights import load_parakeet_tdt  # noqa: E402
from kestrel_kernels.ternary import TernaryLinear  # noqa: E402

DEFAULT_PROBES = ['encoder.layers.0.feed_forward1.linear1', 'encoder.layers.0.feed_forward1.linear2',
                  'encoder.layers.11.self_attn.o_proj', 'encoder.layers.5.conv.pointwise_conv1']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '.cache/parakeet-redux-full')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / '.cache/parakeet-candidates/parakeet-redux-ternary.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--probes', nargs='*', default=DEFAULT_PROBES)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {entry['name']: entry for entry in manifest['quantized_modules']}
    torch.manual_seed(args.seed)

    started = time.time()
    loaded = load_parakeet_tdt(str(args.checkpoint), device='cpu')
    model = loaded.model
    model.eval()
    load_s = time.time() - started

    rows = []
    archive_path = args.checkpoint / 'model.safetensors'
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with safe_open(archive_path, framework='pt') as archive:
        for export_name in args.probes:
            module = model.get_submodule(export_name)
            if not isinstance(module, TernaryLinear):
                raise ValueError(f'{export_name} is {type(module).__name__}, not TernaryLinear')
            entry = entries[export_name]
            qweight = archive.get_tensor(f'{export_name}.qweight')
            scales = archive.get_tensor(f'{export_name}.scales')
            in_features, group_size = entry['in_features'], entry['group_size']
            ours = ternary.decode_matrix(qweight.numpy().tobytes(),
                scales.numpy().astype(np.float32), ternary.TernaryLayout(
                    out_features=entry['out_features'], in_features=in_features,
                    group_size=group_size))
            reference_weight = torch.from_numpy(ours).to(torch.float32)
            official_weight = module._weight.dequantized(torch.float32)
            weight_equal = bool(torch.equal(reference_weight, official_weight))
            weight_diff = (reference_weight.double() - official_weight.double()).abs()
            x = torch.randn(3, in_features, dtype=torch.float32)
            with torch.no_grad():
                reference_out = torch.nn.functional.linear(x, reference_weight)
                official_out = module(x)
                official_again = module(x)
            out_diff = (reference_out.double() - official_out.double()).abs()
            rows.append(dict(
                module=export_name, in_features=in_features, out_features=entry['out_features'],
                group_size=group_size, residentMode=module._weight.mode,
                weightBitExact=weight_equal,
                weightMaxAbsDiff=float(weight_diff.max()), weightMeanAbsDiff=float(weight_diff.mean()),
                forwardBitExact=bool(torch.equal(reference_out, official_out)),
                forwardMaxAbsDiff=float(out_diff.max()),
                repeatDeterministic=bool(torch.equal(official_out, official_again)),
                weightSha256=hashlib.sha256(
                    reference_weight.numpy().tobytes()).hexdigest(),
            ))

    all_exact = all(r['weightBitExact'] and r['forwardBitExact'] and r['repeatDeterministic'] for r in rows)
    receipt = dict(
        status='PASS_PACK_PARITY_BIT_EXACT' if all_exact and rows else 'FAIL_PACK_PARITY',
        scope='kestrel dense-oracle dequantization vs independent ternary.py reference; packed int8 kernel not runnable on this CPU',
        cpu=dict(nativeProbe=kernels_ternary._cpu.ternary_gemm_isa(),
                 note="auto probe 'scalar' (no VNNI on this Skylake-W); set_gemm_isa('avx2') accepted but "
                      "the gemm raises 'no avx2 matrix-multiply path on this machine', so the packed "
                      "compute half of the approved check cannot run on this workstation"),
        archive=dict(path=str(archive_path), sha256=archive_sha),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        seed=args.seed, modelLoadSeconds=round(load_s, 1),
        rows=rows,
        limits=[
            'Validates base-3 unpack, weight rule and scale math both ways; it does not execute the '
            'native int8 gemm8 kernel (impossible on this CPU) or quantized activations.',
            'Fused qkv and encoder-level relative projections are excluded; their rows concatenate.',
            'No decoder/joint/VAD execution, no audio, no tokens, no timing or memory claim.',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], load_s=receipt['modelLoadSeconds'],
        modules=[dict(name=r['module'], weightBitExact=r['weightBitExact'],
                      forwardBitExact=r['forwardBitExact']) for r in rows])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
