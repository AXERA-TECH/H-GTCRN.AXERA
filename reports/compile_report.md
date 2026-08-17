# Compile Report

- Target: AX650
- NPU mode: NPU3
- Pulsar2: 7.0 (5939101d)
- Source ONNX: `/workspace/export/model.onnx`
- AXMODEL: `/workspace/compile/model.axmodel`
- AXMODEL size: 24,198,215 bytes
- MACs: 1,456,526,720
- Quantized output cosine: 0.99876
- Quantized output MSE: 0.00004
- Compile elapsed time: approximately 73 minutes 47 seconds
- Workaround: GRU and Transpose operators use FP32 to avoid the AX650 U8 transpose tiling failure.
- Export fix: ConvTranspose+BN folding now preserves the original PReLU/Tanh activation, matching `GTCRN_IVA.forward()`.
- Build result: success; `error_msg` and `axe_error_msg` are empty.
