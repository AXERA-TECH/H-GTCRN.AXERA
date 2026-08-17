# Run On Board Report

- Board: 10.126.29.50
- Chip: AX650N_CHIP
- Runtime: `/opt/bin/ax_run_model`
- Input: `feat`, shape `(1, 6, 626, 257)`, FP32
- Output: `mask`, shape `(1, 2, 626, 257)`, FP32
- Single-run latency: 19.129 ms
- Python SDK first run: 31.831 ms
- Python SDK 20-run average: 21.142 ms
- Python SDK core RTF: 0.002114
- C++ SDK first run: 22.023 ms
- C++ SDK 20-run average: 20.123 ms
- C++ SDK core RTF: 0.002012
- Python/C++ output cosine: 1.0
- Python/C++ output MAE: 0.0
- Board enhanced audio vs original `infer.py` audio cosine: 0.999468371
- Board enhanced audio vs original `infer.py` audio MAE: 0.000655536079
- Board vs PyTorch mask cosine: 0.99876
- Board vs PyTorch MAE: 0.015434438362717628
- Board vs PyTorch max absolute difference: 0.2780022621154785
- Board vs Pulsar2 simulation cosine: 1.0
- Board vs Pulsar2 simulation MAE: 0.0
- Output finite: true
- Result: pass

The AXMODEL contains the GTCRN neural core (`feat -> mask`). STFT, WPE, IVA,
feature construction, mask application, and ISTFT remain on the CPU side.
