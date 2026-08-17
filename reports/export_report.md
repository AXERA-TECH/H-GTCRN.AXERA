# Export Report

- Model: H-GTCRN neural core split
- ONNX: `model_convert/model.onnx`
- CPU side: STFT/WPE/IVA/feature extraction and mask application/ISTFT
- NPU side: ERB/SFE/Encoder/DPGRNN/Decoder/IERB, input feat -> output mask
- Input: feat[1, 6, 626, 257]
- Output: mask[1, 2, 626, 257]
- Torch-ONNX cosine: 1.00000000
- Calibration: real sample features from 3 repository noisy wavs

## Simplification

- raw size bytes: 443142
- onnxsim size bytes: 301036
- onnxslim size bytes: 325048
- slim ORT cosine: 1.00000012
- canonical ONNX: export/model.onnx
