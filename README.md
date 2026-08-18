# H-GTCRN.AXERA

[H-GTCRN](https://github.com/Max1Wz/H-GTCRN) 语音增强神经核的 AX650/NPU3 导出与量化工程。

本仓库只导出 GTCRN 神经核（`feat → mask`）；STFT/WPE/IVA、特征构造、掩码应用与
ISTFT 保留在 CPU 侧。

## 模型

| 模型 | 输入 | 输出 | Python RTF | C++ RTF |
|------|------|------|:--:|:--:|
| H-GTCRN-core | `feat[1,6,626,257]` | `mask[1,2,626,257]` | 0.0021 | 0.0020 |

| 指标 | 值 |
|------|-----|
| 目标芯片 | AX650/NPU3 |
| ONNX opset | 17 |
| Torch/ONNX cosine | 1.000000119 |
| 量化输出 cosine | 0.99907 |
| 板端 vs PyTorch mask cosine | 0.99876 |
| Python/C++ 输出 cosine | 1.0 |

> RTF = NPU 推理时间 / 10s 音频时长，AX650 板端实测。

## 目录结构

```
h-gtcrn-ax650-export/
├── python/                 # pyaxengine SDK 与示例
├── cpp/                    # C++ SDK（含 download_toolchains.sh）
├── model_convert/          # 导出脚本 + 静态 ONNX + Pulsar2 配置
│   ├── scripts/export_core_onnx.py
│   ├── model.onnx
│   ├── model_meta.json
│   ├── pulsar2_config.json
│   └── compile_pulsar2.sh
├── samples/                # 带噪/增强对比音频
├── reports/                # 导出/编译/仿真/板端报告
└── README.md
```

## 步骤 1：获取原始模型

```bash
git clone --depth 1 https://github.com/Max1Wz/H-GTCRN.git origin/H-GTCRN
```

导出脚本需要：

```text
origin/H-GTCRN/gtcrn_iva.py
origin/H-GTCRN/checkpoints/best_model_0121.tar
origin/H-GTCRN/samples/
```

## 步骤 2：导出 ONNX

```bash
python3 model_convert/scripts/export_core_onnx.py
```

产物：`model_convert/model.onnx`、`model_meta.json`、`sample_input.npy`、
`source_output.npy`、`calib_data/feat.tar.gz`。数据产物由脚本生成、不入库
（`.gitignore` 已排除），重新导出即可复现。

## 步骤 3：生成校准数据

校准特征包 `calib_data/feat.tar.gz` 由上一步自动生成（真实语音特征）。

## 步骤 4：Pulsar2 量化

```bash
bash model_convert/compile_pulsar2.sh
```

产物：`model_convert/output/model.axmodel`。配置要点：U16 图量化，
GRU/Transpose 保持 FP32（规避 AX650 U8 transpose tiling 问题），
`highest_mix_precision=false`。已知可用镜像：

```text
docker-registry.aitsw.axera-tech.com/pulsar2:20260724-temp-5939101d
```

## Python SDK 板端推理

`python/audio_demo.py` 为 wav→wav demo（numpy 实现 CPU 链路，依赖仅 numpy + pyaxengine）：

```bash
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
pip install -r python/requirements.txt

python3 python/audio_demo.py \
  --model model_convert/output/model.axmodel \
  --input-wav samples/Samples1_noisy.wav \
  --output samples/Samples1_board_enhanced.wav
```

core-only 接口（feat → mask）：

```bash
python3 python/h_gtcrn_core_sdk/example.py \
  --model model_convert/output/model.axmodel \
  --input model_convert/sample_input.npy \
  --output-dir output \
  --bench 20 \
  --audio-seconds 10.0
```

## C++ SDK

C++ 二进制支持 **wav → wav 端到端**（CPU: STFT/WPE/IVA/ISTFT，NPU: GTCRN 核），
也保留原 feat.bin → mask.bin 的 core-only 模式。

交叉编译工具链一键下载（Arm GNU 9.2 + AX650 BSP，放到 `cpp/toolchains/`，不入库）：

```bash
bash cpp/download_toolchains.sh
bash cpp/build_ax650.sh        # -> cpp/bin/h_gtcrn_ax650
```

板端运行（wav→wav，16kHz PCM16，1/2 声道，长度 ≤ 10.0s 自动补零）：

```bash
./cpp/bin/h_gtcrn_ax650 \
  model_convert/output/model.axmodel \
  samples/Samples1_noisy.wav \
  output_enhanced.wav \
  --bench 20 \
  --audio-seconds 10.0
```

core-only 模式（原接口）：`./cpp/bin/h_gtcrn_ax650 model.axmodel sample_input.bin output --bench 20`。
`sample_input.bin` 由 `sample_input.npy` 转换而来（`numpy tofile`），也可从
[H-GTCRN.AXERA HuggingFace](https://huggingface.co/AXERA-TECH/H-GTCRN.AXERA)
仓库 `examples/` 获取。板端运行库在 `/soc/lib`，系统已配置搜索路径，直接运行
即可；如提示找不到 `libax_*.so` 再加
`export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}`。详见 `cpp/README.md`。

## 示例音频

- `samples/Samples1_noisy.wav` — 原始 16kHz 双通道带噪输入（RMS 0.09499）
- `samples/Samples1_board_enhanced.wav` — AX650 板端增强输出（RMS 0.02932）
- `samples/Samples1_core_ref_enhanced.wav` — ONNX 参考增强输出

## 报告

`reports/` 下有完整的导出/编译/仿真/板端/性能/自测报告。

## 参考

- [H-GTCRN](https://github.com/Max1Wz/H-GTCRN) — 原始模型
- [H-GTCRN.AXERA（HuggingFace 预编译模型+SDK）](https://huggingface.co/AXERA-TECH/H-GTCRN.AXERA)
- [Magnetar](https://github.com/AXERA-TECH/Magnetar) — AXERA 模型部署工具链
