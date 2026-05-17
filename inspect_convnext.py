import torch
from pathlib import Path
p = Path("best_convnext.pth")
print("FILE:", p.resolve())
ckpt = torch.load(p, map_location='cpu')
print("TYPE:", type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("TOP_KEYS:", keys[:50])
    sd = None
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        # Heuristic: if all values are tensors, treat as state_dict
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
    if sd is None:
        print("NO_STATE_DICT_FOUND")
    else:
        print("STATE_DICT_LEN:", len(sd))
        for i, (k, v) in enumerate(list(sd.items())[:60]):
            try:
                shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)
            except Exception:
                shape = type(v)
            print(f"SD[{i}]: {k} -> {shape}")
        # Count blocks per stage
        import re
        stages = {}
        for k in sd.keys():
            m = re.match(r'stages\.(\d+)\.blocks\.(\d+)\.', k)
            if m:
                si = int(m.group(1)); bi = int(m.group(2))
                stages.setdefault(si, set()).add(bi)
        for si in sorted(stages.keys()):
            print(f"STAGE_{si}_BLOCK_COUNT:", len(stages[si]))
        # List potential classifier/head tensors
        for k, v in sd.items():
            lk = k.lower()
            if any(tok in lk for tok in ['head', 'classifier', 'fc', 'norm', 'global_pool']):
                try:
                    shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)
                except Exception:
                    shape = type(v)
                print(f"CANDIDATE_HEAD_TENSOR: {k} -> {shape}")
        # Find classifier-ish weights
        for k, v in sd.items():
            if any(tok in k.lower() for tok in ['head', 'fc', 'classifier', 'norm', 'patch_embed']):
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    print("LIKELY_CLASSIFIER_WEIGHT:", k, tuple(v.shape))
                    break
    # metadata heuristics
    for meta in ['epoch', 'epochs', 'best_epoch', 'num_classes', 'image_size', 'img_size', 'input_size', 'cfg', 'config', 'args']:
        if meta in ckpt:
            print(f"META_{meta.upper()}:", ckpt[meta])
    if 'optimizer' in ckpt or 'optimizer_state_dict' in ckpt:
        print('OPTIMIZER_STATE_PRESENT')
else:
    print('SAVED_OBJECT_ATTRS:', [a for a in dir(ckpt) if not a.startswith('_')][:200])
print('INSPECTION_DONE')
