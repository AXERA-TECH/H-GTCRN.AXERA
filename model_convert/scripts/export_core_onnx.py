#!/usr/bin/env python3
import json
import copy
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_CONVERT_DIR = PROJECT_DIR / "model_convert"
REPO_DIR = PROJECT_DIR / "origin" / "H-GTCRN"
EXPORT_DIR = MODEL_CONVERT_DIR
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from gtcrn_iva import GTCRN_IVA, auxiva, fd_wpe, multi_channel_stft  # noqa: E402


class GTCRNCore(nn.Module):
    """Trace-friendly neural core: feature tensor -> complex ratio mask."""

    def __init__(self, full: GTCRN_IVA):
        super().__init__()
        self.erb = full.erb
        self.sfe = full.sfe
        self.encoder = full.encoder
        self.dpgrnn1 = full.dpgrnn1
        self.dpgrnn2 = full.dpgrnn2
        self.decoder = full.decoder

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self.erb.bm(feat)
        feat = self.sfe(feat)
        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)
        m_feat = self.decoder(feat, en_outs)
        return self.erb.bs(m_feat)


def fold_convtranspose_bn(deconv: nn.ConvTranspose2d, bn: nn.BatchNorm2d) -> nn.ConvTranspose2d:
    folded = copy.deepcopy(deconv).eval()
    weight = folded.weight.detach().clone()
    bias = folded.bias.detach().clone() if folded.bias is not None else torch.zeros(
        folded.out_channels, dtype=weight.dtype, device=weight.device
    )

    groups = folded.groups
    in_per_group = folded.in_channels // groups
    out_per_group = folded.out_channels // groups
    weight = weight.view(groups, in_per_group, out_per_group, *weight.shape[2:])

    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    shift = bn.bias.detach() - bn.running_mean.detach() * scale
    scale = scale.view(groups, 1, out_per_group, 1, 1)
    weight = weight * scale
    weight = weight.view_as(folded.weight)
    bias = bias * scale.reshape(-1) + shift

    folded.weight = nn.Parameter(weight)
    folded.bias = nn.Parameter(bias)
    return folded


class FoldedConvTranspose2d(nn.Module):
    def __init__(self, deconv: nn.ConvTranspose2d, bn: nn.BatchNorm2d):
        super().__init__()
        self.deconv = fold_convtranspose_bn(deconv, bn)

    def forward(self, x):
        return self.deconv(x)


class FlattenLayerNorm(nn.Module):
    """把多维 LayerNorm 展平为最后一维，只保留 axis=-1。"""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.shape = tuple(ln.normalized_shape)
        self.flat = int(np.prod(self.shape))
        self.ln = nn.LayerNorm(self.flat, eps=ln.eps, elementwise_affine=True)
        with torch.no_grad():
            self.ln.weight.copy_(ln.weight.reshape(-1))
            self.ln.bias.copy_(ln.bias.reshape(-1))

    def forward(self, x):
        prefix = x.shape[:-len(self.shape)]
        x = x.reshape(*prefix, self.flat)
        x = self.ln(x)
        return x.reshape(*prefix, *self.shape)


def load_full_model() -> GTCRN_IVA:
    model = GTCRN_IVA().eval()
    ckpt = torch.load(REPO_DIR / "checkpoints" / "best_model_0121.tar", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def load_wav(path: Path) -> torch.Tensor:
    noisy, fs = sf.read(path, dtype="float32")
    if fs != 16000:
        raise ValueError(f"{path}: expected 16000 Hz, got {fs}")
    if noisy.ndim == 1:
        return torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
    return torch.from_numpy(noisy.T).unsqueeze(0)


def compute_feature(full: GTCRN_IVA, x: torch.Tensor) -> torch.Tensor:
    stft_kwargs = {
        "n_fft": full.n_fft,
        "hop_length": full.hop_len,
        "win_length": full.win_len,
        "window": torch.hann_window(full.win_len),
        "onesided": True,
    }
    spec_orig = multi_channel_stft(x.transpose(1, 2), **stft_kwargs)
    spec_drb = fd_wpe(spec_orig, rt60=0.3, shift=256, D=2, fs=16000, num_iter=1)
    spec_2ch = auxiva(spec_drb.transpose(1, 3), n_iter=10).transpose(1, 3)

    spec_norm = torch.norm(spec_2ch, dim=(2, 3))
    pred = torch.where(spec_norm[:, 0] < spec_norm[:, 1], 1, 0)
    pred = pred.view(-1, 1, 1, 1)
    spec_selected = spec_2ch[:, 0] * pred[:, 0] + spec_2ch[:, 1] * (1 - pred[:, 0])
    spec_unselected = spec_2ch[:, 1] * pred[:, 0] + spec_2ch[:, 0] * (1 - pred[:, 0])

    spec_sel = torch.view_as_real(spec_selected).permute(0, 3, 2, 1)
    spec_sel_mag = torch.norm(spec_sel, dim=1, keepdim=True).clamp(1e-12)
    spec_sel_log = torch.log10(spec_sel_mag)

    spec_un = torch.view_as_real(spec_unselected).permute(0, 3, 2, 1)
    spec_un_mag = torch.norm(spec_un, dim=1, keepdim=True).clamp(1e-12)
    spec_un_log = torch.log10(spec_un_mag)

    spec = torch.view_as_real(spec_orig)
    spec = spec.permute(0, 1, 4, 3, 2).reshape(spec.shape[0], -1, spec.shape[3], spec.shape[2])
    return torch.cat([spec, spec_sel_log, spec_un_log], dim=1).contiguous()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def write_calibration(samples: list[np.ndarray]) -> None:
    calib_dir = EXPORT_DIR / "calib_data" / "feat"
    calib_dir.mkdir(parents=True, exist_ok=True)
    for idx, arr in enumerate(samples):
        np.save(calib_dir / f"{idx:04d}.npy", np.ascontiguousarray(arr, dtype=np.float32))
    tar_path = EXPORT_DIR / "calib_data" / "feat.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for npy in sorted(calib_dir.glob("*.npy")):
            tar.add(npy, arcname=npy.name)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    full = load_full_model()
    core = GTCRNCore(full).eval()
    core.decoder.de_convs[3] = FoldedConvTranspose2d(full.decoder.de_convs[3].conv, full.decoder.de_convs[3].bn)
    core.decoder.de_convs[4] = FoldedConvTranspose2d(full.decoder.de_convs[4].conv, full.decoder.de_convs[4].bn)
    core.dpgrnn1.intra_ln = FlattenLayerNorm(full.dpgrnn1.intra_ln)
    core.dpgrnn1.inter_ln = FlattenLayerNorm(full.dpgrnn1.inter_ln)
    core.dpgrnn2.intra_ln = FlattenLayerNorm(full.dpgrnn2.intra_ln)
    core.dpgrnn2.inter_ln = FlattenLayerNorm(full.dpgrnn2.inter_ln)

    wavs = sorted((REPO_DIR / "samples").glob("Samples*/Samples*_noisy.wav"))
    if not wavs:
        raise FileNotFoundError("no sample wavs found")

    feats = []
    with torch.inference_mode():
        for wav in wavs:
            feats.append(compute_feature(full, load_wav(wav)).detach().cpu().numpy().astype(np.float32))
    feat = torch.from_numpy(feats[0])

    onnx_path = EXPORT_DIR / "model.onnx"
    with torch.inference_mode():
        ref = core(feat).detach().cpu().numpy().astype(np.float32)

    torch.onnx.export(
        core,
        (feat,),
        str(onnx_path),
        input_names=["feat"],
        output_names=["mask"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"feat": feats[0]})[0].astype(np.float32)
    cos = cosine(ref, ort_out)

    np.save(EXPORT_DIR / "sample_input.npy", feats[0])
    np.save(EXPORT_DIR / "source_output.npy", ref)
    write_calibration(feats)

    meta = {
        "model_name": "H-GTCRN-core",
        "framework": "pytorch",
        "route": "general_split",
        "split_note": "CPU computes STFT/WPE/IVA/features and ISTFT; AXMODEL runs GTCRN neural core feat->mask.",
        "inputs": [{"name": "feat", "shape": list(feats[0].shape), "dtype": "float32", "layout": "NCHW"}],
        "outputs": [{"name": "mask", "shape": list(ref.shape), "dtype": "float32"}],
        "opset": 17,
        "torch_onnx_cosine": cos,
    }
    (EXPORT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPORT_DIR / "export_report.md").write_text(
        "\n".join(
            [
                "# Export Report",
                "",
                "- Model: H-GTCRN neural core split",
                f"- ONNX: {onnx_path}",
                "- CPU side: STFT/WPE/IVA/feature extraction and mask application/ISTFT",
                "- NPU side: ERB/SFE/Encoder/DPGRNN/Decoder/IERB, input feat -> output mask",
                f"- Input: feat{list(feats[0].shape)}",
                f"- Output: mask{list(ref.shape)}",
                f"- Torch-ONNX cosine: {cos:.8f}",
                f"- Calibration: real sample features from {len(feats)} repository noisy wavs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if cos < 0.99:
        raise RuntimeError(f"Torch-ONNX cosine {cos:.8f} < 0.99")
    print(json.dumps({"onnx": str(onnx_path), "cosine": cos, "input_shape": list(feats[0].shape)}, indent=2))


if __name__ == "__main__":
    main()
