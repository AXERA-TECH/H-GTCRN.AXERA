# H-GTCRN-core C++ SDK

- 输入（与 model_meta.json 一致）: feat[1, 6, 626, 257]
- 输出（与 model_meta.json 一致）: mask[1, 2, 626, 257]
- 直接链接 AX Engine runtime（`ax_engine`/`ax_sys`），目标: AX650

## 交叉编译工具链

一键下载（Arm GNU 9.2 交叉编译器 + AX650 BSP SDK，放到 `toolchains/`，不入库）：

```bash
bash download_toolchains.sh
```

或手动准备（已装系统 `aarch64-linux-gnu-g++` 或已有工具链时）：

```bash
wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
export TOOLCHAIN_ROOT=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
export BSP_MSP_DIR=$(pwd)/ax650n_bsp_sdk/msp/out
```

## 编译

```bash
bash build_ax650.sh        # -> bin/h_gtcrn_ax650
```

或手动 CMake：

```bash
cmake -S . -B build_ax650 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++ \
  -DBSP_MSP_DIR=$BSP_MSP_DIR
cmake --build build_ax650
```

板端 native 编译（板上有 cmake 时）可用 `-DAX_RUNTIME_ROOT=/soc` 替代
`BSP_MSP_DIR`，头文件/库直接取自设备运行时。

## 板端运行

wav → wav 端到端（16kHz PCM16，1/2 声道，长度 ≤ 10.0s 自动补零）：

```bash
LD_LIBRARY_PATH=/soc/lib ./bin/h_gtcrn_ax650 \
  models/model.axmodel \
  samples/Samples1_noisy.wav \
  output_enhanced.wav \
  --bench 20 \
  --audio-seconds 10.0
```

core-only 模式（原接口，feat.bin → mask.bin）：

```bash
LD_LIBRARY_PATH=/soc/lib ./bin/h_gtcrn_ax650 \
  models/model.axmodel \
  examples/sample_input.bin \
  cpp_output \
  --bench 20 \
  --audio-seconds 10.0
```

CPU 链路（`src/audio_chain.hpp`，与 python/audio_demo.py 逐式对齐）：
STFT → WPE → auxIVA → 特征构造 → [NPU] → 掩码应用 → ISTFT。
AX650 板端实测：C++ 输出与 torch 链参考 cosine 0.99999，与 Python SDK 输出
cosine 1.0；core RTF 0.0020，CPU 链路约 9.5s/10s 音频（WPE 统计为整段算法）。
