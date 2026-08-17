#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="docker-registry.aitsw.axera-tech.com/pulsar2:20260724-temp-5939101d"

docker run --rm --init \
  -e TMPDIR=/workspace/.pulsar_tmp \
  -v "$PACKAGE_ROOT:/workspace" \
  "$IMAGE" pulsar2 build \
  --config /workspace/model_convert/pulsar2_config.json
