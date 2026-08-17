# Package Self-Test

- Static package checks: pass
- Python SDK import: pass
- C++ CMake configure: pass
- Board `setup.sh`: pass using `/root/miniforge3/bin/python`
- Board `run.sh`: pass using `AxEngineExecutionProvider`
- Package output shape: `(1, 2, 626, 257)`
- Package output/reference mask cosine: 0.99876
- Package output/reference mask MSE: 0.00004
- ONNX enhanced wav vs original `infer.py` wav cosine: 0.9999999998
- Board enhanced wav vs original `infer.py` wav cosine: 0.999468371
- Board enhanced wav vs original `infer.py` wav MAE: 0.000655536079
