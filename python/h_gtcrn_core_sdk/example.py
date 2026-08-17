import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h_gtcrn_core_sdk.inference import ModelSession
from h_gtcrn_core_sdk.postprocess import postprocess
from h_gtcrn_core_sdk.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description="H-GTCRN-core inference example")
    parser.add_argument("--model", required=True, help="AXMODEL 路径")
    parser.add_argument("--input", nargs="+", required=True, help="输入 npy（每输入一个，与 model_meta 顺序一致）")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--bench", type=int, default=1, help="统计推理次数")
    parser.add_argument("--audio-seconds", type=float, default=10.0, help="用于 RTF 统计的音频时长")
    args = parser.parse_args()

    arrays = [np.load(p).astype(np.float32) for p in args.input]
    session = ModelSession(args.model)
    feeds = preprocess(*arrays)
    start = time.perf_counter()
    raw = session.run_named(feeds)
    first_ms = (time.perf_counter() - start) * 1000.0
    bench_ms = first_ms
    if args.bench > 1:
        start = time.perf_counter()
        for _ in range(args.bench):
            raw = session.run_named(feeds)
        bench_ms = (time.perf_counter() - start) * 1000.0 / args.bench
    result = postprocess(raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(raw.values()):
        np.save(out_dir / f"output_{i}.npy", np.asarray(arr, dtype=np.float32))
    try:
        json.dumps(result)
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except TypeError:
        np.save(out_dir / "result.npy", np.asarray(result, dtype=np.float32))

    print("backend:", session.backend)
    print("inputs:", session.input_names)
    print("outputs:", session.output_names)
    print("first_run_ms: %.3f" % first_ms)
    print("bench_avg_ms: %.3f" % bench_ms)
    print("core_rtf: %.6f" % (bench_ms / 1000.0 / args.audio_seconds))
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
