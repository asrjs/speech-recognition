#!/usr/bin/env python
"""Probe NeMo joint to understand its architecture."""
import torch
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

print("Loading NeMo...")
nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu"
)
nemo.eval()
joint = nemo.joint
print(f"Joint type: {type(joint).__name__}")
print(f"Joint attributes:")
for attr in dir(joint):
    if not attr.startswith("_"):
        try:
            val = getattr(joint, attr)
            if isinstance(val, torch.nn.Module):
                print(f"  {attr}: {type(val).__name__}")
                for sub_attr in dir(val):
                    if not sub_attr.startswith("_"):
                        try:
                            sub_val = getattr(val, sub_attr)
                            if isinstance(sub_val, torch.nn.Module):
                                print(f"    {sub_attr}: {type(sub_val).__name__}")
                        except Exception:
                            pass
        except Exception:
            pass

# Get enc projection
print("\njoint.enc:", joint.enc)
print("joint.pred:", joint.pred)

# Test joint_after_projection manually
print("\n--- Test joint_after_projection with random inputs ---")
import torch
f = torch.randn(1, 640)  # encoder projection
g = torch.randn(1, 640)  # decoder projection
out = joint.joint_after_projection(f, g)
print(f"joint_after_projection output shape: {out.shape}, mean: {float(out.mean()):.4f}")