        # H-GTCRN-core Python SDK

        - 输入（与 model_meta.json 一致）: feat[1, 6, 626, 257]
        - 输出（与 model_meta.json 一致）: mask[1, 2, 626, 257]
        - 预处理: CPU 侧完成 STFT/WPE/IVA 和特征构造；SDK 接收 feat FP32 张量。
        - 后处理: 返回复数比例 mask；CPU 侧负责应用 mask 和 ISTFT。
        - 示例输入: examples/sample_input.npy

> NPU 专用发布版：端到端 NPU 验证已通过，SDK 仅依赖 pyaxengine（无 onnxruntime/torch/transformers 回退）。


        ```bash
        LD_LIBRARY_PATH=/soc/lib PYTHONPATH=$PWD/python python3 h_gtcrn_core_sdk/example.py           --model models/model.axmodel --input input.npy --output-dir output
        ```
