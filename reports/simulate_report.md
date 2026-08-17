# Simulate Report

- Method: Pulsar2 run
- Input: `feat`, shape `(1, 6, 626, 257)`, FP32
- Output: `mask`, shape `(1, 2, 626, 257)`, FP32
- Cosine similarity: 0.99876
- MSE: 0.00004
- Source: fixed rebuild log after preserving ConvTranspose+BN activation
- Output finite: true
- Output nonzero elements: 320001
- Result: pass (cosine >= 0.99)
