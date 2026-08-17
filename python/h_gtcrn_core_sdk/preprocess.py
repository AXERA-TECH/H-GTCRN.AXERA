import numpy as np

def preprocess(feat):
    return [np.ascontiguousarray(feat, dtype=np.float32)]
