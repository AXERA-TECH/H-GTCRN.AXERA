#!/usr/bin/env bash
# 下载交叉编译器与 AXERA BSP SDK（默认放到 cpp/toolchains/，不随 git 提交）。
#
# 用法:
#   bash cpp/download_toolchains.sh            # gcc + AX650 BSP
#
# 若已把工具链放在其他目录，可导出环境变量后直接编译：
#   export TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
#   export BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out
#   bash cpp/build_ax650.sh
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAINS_DIR="${CPP_DIR}/toolchains"

GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
GCC_DIR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

# gcc 交叉编译器
if [[ -x "${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++" ]]; then
  echo "Found ${TOOLCHAINS_DIR}/${GCC_DIR}"
else
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Downloading ${GCC_URL}"
  wget -q --show-progress "${GCC_URL}" -O gcc.tar.xz
  tar -xf gcc.tar.xz
  rm -f gcc.tar.xz
  echo "Done: ${TOOLCHAINS_DIR}/${GCC_DIR}"
fi

# AX650 BSP SDK（含 ax_engine/ax_sys 头文件与库）
if [[ -f "${TOOLCHAINS_DIR}/ax650n_bsp_sdk/msp/out/include/ax_engine_api.h" ]]; then
  echo "Found ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
else
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Cloning https://github.com/AXERA-TECH/ax650n_bsp_sdk.git"
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git --depth=1
  echo "Done: ${TOOLCHAINS_DIR}/ax650n_bsp_sdk"
fi
