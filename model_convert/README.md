# Model Conversion

The included ONNX is static shape and uses AX650/NPU3. Rebuild from the package
root with:

```bash
bash model_convert/compile_pulsar2.sh
```

The result is written to `model_convert/output/model.axmodel`. GRU and
Transpose operators remain FP32 to avoid the AX650 U8 transpose tiling error
observed for the `(626, 33, 8)` tensor.
