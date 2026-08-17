import numpy as np

def postprocess(mask):
    if isinstance(mask, dict):
        mask = mask["mask"]
    return np.asarray(mask, dtype=np.float32)
