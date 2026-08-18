#!/usr/bin/env python3
"""H-GTCRN 板端 wav→wav 降噪 demo（numpy 实现，依赖仅 numpy + pyaxengine）。

CPU 链路（与官方 GTCRN_IVA 一致）：STFT → WPE 去混响 → auxIVA 分离 →
通道选择 + 特征构造 → [NPU: GTCRN 核] → 复数掩码应用 → ISTFT。
仅支持 16kHz PCM16 wav；长度 ≤ 10.0s（626 帧）自动补零，超过报错。
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h_gtcrn_core_sdk import ModelSession

N_FFT = 512
HOP = 256
WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float64)  # 等价 torch.hann_window(512, periodic=True)
N_FREQ = N_FFT // 2 + 1          # 257
MAX_FRAMES = 626                 # feat[1,6,626,257]，约 10.0s @16kHz
SR = 16000


def read_wav(path):
    """读 PCM16 wav → [ch, L] float64；单声道复制为双声道。"""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"expected 16000 Hz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "only 16-bit PCM wav supported"
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).T
    if ch == 1:
        x = np.stack([x[0], x[0]], axis=0)
    elif ch > 2:
        x = x[:2]
    return np.ascontiguousarray(x)


def write_wav(path, x, sr=SR):
    pcm = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())


def stft(x):
    """x [ch, L] → [ch, F, T] complex（center=True 反射填充）。"""
    ch, L = x.shape
    xp = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    T = (xp.shape[1] - N_FFT) // HOP + 1
    frames = np.stack([xp[:, t * HOP:t * HOP + N_FFT] for t in range(T)], axis=0)  # [T,ch,N]
    return np.fft.rfft(frames * WIN, axis=2).transpose(1, 2, 0)  # [ch,F,T]


def istft(spec):
    """spec [F, T] complex → (T-1)*hop + n_fft 长度信号，裁掉 center 填充。"""
    F, T = spec.shape
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * WIN  # [T,N]
    out_len = (T - 1) * HOP + N_FFT
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    for t in range(T):
        s = t * HOP
        out[s:s + N_FFT] += frames[t]
        wsum[s:s + N_FFT] += WIN * WIN
    wsum[wsum < 1e-10] = 1.0
    out = out / wsum
    pad = N_FFT // 2
    return out[pad:out_len - pad]  # center 填充裁剪（对齐 torch.istft(center=True)）


def fd_wpe(X, rt60=0.3, shift=256, D=2, fs=16000, num_iter=1):
    """WPE 去混响。X [M,F,T] complex → [M,F,T] complex（与 torch 版逐式对齐）。"""
    M, F, T = X.shape
    eps = 1e-3 * np.mean(np.max(np.max(np.abs(X) ** 2, axis=-1), axis=0))
    Lg = int(rt60 * fs / shift)
    Xp = X.transpose(1, 0, 2)                       # [F,M,T]
    X_delay = np.zeros((F, M * Lg, T), dtype=X.dtype)
    for l in range(Lg):
        X_delay[:, l * M:(l + 1) * M, D + l:T] = Xp[:, :, 0:T - D - l]
    Y = Xp.copy()
    for _ in range(num_iter):
        lambdaa = np.maximum(np.mean(np.abs(Y) ** 2, axis=-2, keepdims=True), eps)  # [F,1,T]
        temp = X_delay / lambdaa
        R = temp @ np.conj(X_delay.transpose(0, 2, 1))   # [F,36,36]
        P = temp @ np.conj(Xp.transpose(0, 2, 1))        # [F,36,M]
        G = np.linalg.solve(R + eps * np.eye(M * Lg), P)  # [F,36,M]
        Y = Xp - np.conj(G.transpose(0, 2, 1)) @ X_delay
    return Y.transpose(1, 0, 2)                     # [M,F,T]


def auxiva(X, n_src=2, n_iter=10, proj_back=True, model="laplace"):
    """auxIVA 声源分离。X [T,F,M] complex → [T,F,M] complex（与 torch 版逐式对齐）。"""
    T, F, M = X.shape
    eps = 1e-10
    eyes = np.eye(M, dtype=X.dtype)
    W = np.zeros((F, M, M), dtype=X.dtype)
    W[:, :, :] = np.tile(np.eye(n_src, dtype=X.dtype), (F, 1, 1))[:, :n_src, :]
    Xp = X.transpose(1, 2, 0).copy()                # [F,M,T]
    Y = np.zeros((F, M, T), dtype=X.dtype)
    r = np.zeros((M, T), dtype=np.float32)
    for _ in range(n_iter):
        Y = W @ Xp                                  # [F,M,T]
        if model == "laplace":
            r = 2.0 * np.sqrt((Y.real ** 2 + Y.imag ** 2).sum(axis=0))  # [M,T]，沿 F 求和
        else:
            r = (Y.real ** 2 + Y.imag ** 2).sum(axis=0) / F  # 沿 F 求和
        r[r < eps] = eps
        r_inv = 1.0 / r                             # [M,T]
        for s in range(n_src):
            V = ((Xp * r_inv[None, s, None, :]) @ np.conj(Xp.transpose(0, 2, 1))) / T  # [F,M,M]
            WV = W @ V
            e_s = np.tile(eyes[:, s], (F, 1))[:, :, None]   # [F,M,1]
            W[:, s, :] = np.conj(np.linalg.solve(WV + eps * eyes, e_s))[:, :, 0]
            denom = (W[:, None, s, :] @ V @ np.conj(W[:, s, :, None]))[:, :, 0]
            W[:, s, :] /= np.sqrt(denom + eps)
    Y = W @ Xp                                     # [F,M,T]
    if proj_back:
        # ref = X 的第 0 通道 [T,F]；对 T（帧）求和 → 每个 (f, 声源) 一个缩放系数
        ref = X[:, :, 0]                           # [T,F]
        num = np.sum(np.conj(ref).T[:, None, :] * Y, axis=2)    # [F,M]
        denom = np.sum(np.abs(Y) ** 2, axis=2)                  # [F,M]
        c = np.ones((F, M), dtype=X.dtype)
        valid = denom > 0.0
        c[valid] = num[valid] / denom[valid]
        Y *= np.conj(c)[:, :, None]
    return Y.transpose(2, 0, 1)                    # [T,F,M]


def main():
    parser = argparse.ArgumentParser(description="H-GTCRN AX650 板端 wav→wav 降噪（numpy 版）。")
    parser.add_argument("--model", default="../models/model.axmodel")
    parser.add_argument("--input-wav", default="../samples/Samples1_noisy.wav")
    parser.add_argument("--output", default="../samples/Samples1_board_enhanced.wav")
    args = parser.parse_args()

    x = read_wav(args.input_wav)                   # [M,L]
    L = x.shape[1]
    T = L // HOP + 1
    if T > MAX_FRAMES:
        raise ValueError(f"input too long: {T} frames > {MAX_FRAMES} (约 10.0s @16kHz)")
    if T < MAX_FRAMES:
        # 补零到 626 帧（零帧对 WPE/IVA 统计无贡献，proj_back 缩放不变性保证结果一致）
        x = np.pad(x, ((0, 0), (0, (MAX_FRAMES - T) * HOP)))
        T = MAX_FRAMES

    spec_orig = stft(x)                            # [M,F,T]
    spec_drb = fd_wpe(spec_orig)                   # WPE
    spec_2ch = auxiva(spec_drb.transpose(2, 1, 0)).transpose(2, 1, 0)  # [M,F,T]

    # 通道选择：能量较小的通道
    spec_norm = np.sqrt((spec_2ch.real ** 2 + spec_2ch.imag ** 2).sum(axis=(1, 2)))  # [M]
    pred = 1 if spec_norm[0] < spec_norm[1] else 0
    spec_selected = spec_2ch[pred]                 # [F,T]
    spec_unselected = spec_2ch[1 - pred]

    sel_log = np.log10(np.maximum(np.abs(spec_selected), 1e-12)).T[None]   # [1,T,F]
    un_log = np.log10(np.maximum(np.abs(spec_unselected), 1e-12)).T[None]

    spec = np.stack([spec_orig[0].real, spec_orig[0].imag,
                     spec_orig[1].real, spec_orig[1].imag], axis=0).transpose(0, 2, 1)  # [4,T,F]
    feat = np.concatenate([spec, sel_log, un_log], axis=0)[None]         # [1,6,T,F]
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    session = ModelSession(args.model)
    mask = session.run_named({"feat": feat})["mask"]                     # [1,2,T,F]
    m = mask[0]                                                          # [2,T,F]

    s_real = spec[0] * m[0] - spec[1] * m[1]
    s_imag = spec[1] * m[0] + spec[0] * m[1]
    spec_enh = (s_real + 1j * s_imag).T                                        # [F,T]
    out = istft(spec_enh)[:L]
    write_wav(args.output, out)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
