import torch

ckpt = torch.load(
    "transformer_feat_varlen_scratch/best_model.pth",
    map_location="cpu",
    weights_only=False,
)

# 处理不同的保存格式
if isinstance(ckpt, dict):
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
else:
    sd = ckpt

print(f"Total keys: {len(sd)}")
print("\nAll keys:")
for i, k in enumerate(sorted(sd.keys())):
    print(f"{i+1}. {k}: {sd[k].shape if hasattr(sd[k], 'shape') else type(sd[k])}")

