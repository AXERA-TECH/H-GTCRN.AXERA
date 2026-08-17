import numpy as np

DEFAULT_PROVIDER = "AxEngineExecutionProvider"


class ModelSession:
    """NPU 专用推理会话：仅支持 AX 芯片端到端运行（pyaxengine），无 CPU/onnxruntime 回退。

    input/output 名称与 export/model_meta.json 一致（AXMODEL 即按此编译）。
    非 AX 环境（缺少 pyaxengine 或 AX provider）时直接报错，不做 CPU 兜底。
    """

    def __init__(self, model_path, providers=None):
        try:
            import axengine as axe
        except ImportError as exc:
            raise RuntimeError(
                "SDK 为 NPU 专用发布版，仅支持在 AX 芯片上运行；请先安装 requirements.txt "
                "并在板端执行（无 onnxruntime/torch/transformers 回退）"
            ) from exc
        self.session = axe.InferenceSession(
            model_path, providers=providers or [DEFAULT_PROVIDER])
        self.backend = "axengine"
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run_named(self, feeds, names=None):
        """Run inference and return a dict keyed by output tensor name."""
        names = names or self.input_names
        if isinstance(feeds, dict):
            feed = {
                name: np.ascontiguousarray(feeds[name], dtype=np.float32)
                for name in names
            }
        else:
            if len(feeds) != len(names):
                raise ValueError(f"输入数量不匹配: {len(feeds)} != {len(names)}")
            feed = {
                name: np.ascontiguousarray(arr, dtype=np.float32)
                for name, arr in zip(names, feeds)
            }
        outputs = self.session.run(None, feed)
        return {name: value for name, value in zip(self.output_names, outputs)}
