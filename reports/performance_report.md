# Performance Report

- Board: AX650N (`10.126.29.50`)
- Runtime: AX Engine 2.12.0s / pyaxengine `AxEngineExecutionProvider`
- Model: H-GTCRN neural core (`feat -> mask`)
- Sample duration for RTF: 10.0 s
- Input tensor: `feat[1, 6, 626, 257]`, FP32
- Output tensor: `mask[1, 2, 626, 257]`, FP32

| Runtime | First run | 20-run average | Core RTF |
|---|---:|---:|---:|
| Python SDK | 31.831 ms | 21.142 ms | 0.002114 |
| C++ SDK | 22.023 ms | 20.123 ms | 0.002012 |

| Check | Value |
|---|---:|
| Python/C++ output cosine | 1.0 |
| Python/C++ output MAE | 0.0 |
| Python/C++ max absolute difference | 0.0 |
| Board/reference mask cosine | 0.99876 |
| Board/original infer audio cosine | 0.999468371 |
| Board/original infer audio MAE | 0.000655536079 |
| ONNX/original infer audio cosine | 0.9999999998 |
| ONNX/original infer audio MAE | 0.0000000109 |

Audio sample:

- Input: `samples/Samples1_noisy.wav`, 16 kHz, 10.0 s, 2 channels, RMS 0.09499
- Original `infer.py` output RMS: 0.02980
- Board output: `samples/Samples1_board_enhanced.wav`, RMS 0.02932

The RTF number covers only the compiled NPU neural core. CPU-side STFT/WPE/IVA,
feature construction, mask application, and ISTFT are not included.

The export was fixed on 2026-08-13: `FoldedConvTranspose2d` now preserves the
original activation (`PReLU`/`Tanh`) after folding BatchNorm into ConvTranspose.
Before this fix, ONNX and AXMODEL were mutually consistent but did not match the
original `GTCRN_IVA.forward()` audio.
