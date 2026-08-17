#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h_gtcrn_core_sdk import ModelSession


def multi_channel_stft(x, n_fft, hop_length, win_length, window, onesided=True):
    bs = x.shape[0]
    x = x.transpose(1, 2).reshape(-1, x.size(1))
    x = torch.stft(x, n_fft, hop_length, win_length, window, onesided=onesided, return_complex=True)
    x = torch.view_as_real(x)
    x = x.view(bs, -1, x.shape[1], x.shape[2], 2)
    return torch.complex(x[..., 0], x[..., 1])


def fd_wpe(X, rt60, shift, D=2, fs=16000, num_iter=1):
    B, M, F, T = X.shape
    device = X.device
    eps = 1e-3 * torch.mean(torch.max(torch.max(X.abs() ** 2, dim=-1).values, dim=-2).values).to(device)
    Lg = int(rt60 * fs / shift)
    Xp = torch.permute(X, [0, 2, 1, 3])
    eyes = torch.tile(torch.eye(M * Lg, M * Lg, dtype=X.dtype), (B, F, 1, 1)).to(device)
    X_delay = torch.zeros(B, F, M * Lg, T, dtype=X.dtype).to(device)
    for l in range(Lg):
        X_delay[:, :, l * M : (l + 1) * M, D + l : T] = Xp[:, :, :, 0 : T - D - l]
    Y = Xp.clone().to(device)
    for _ in range(num_iter):
        lambdaa = torch.max(torch.mean(torch.abs(Y) ** 2, dim=-2, keepdim=True), eps).to(device)
        temp = X_delay / lambdaa
        R = temp @ torch.conj(X_delay.transpose(-2, -1))
        P = temp @ torch.conj(Xp.transpose(-2, -1))
        G = torch.linalg.inv(R + eps * eyes) @ P
        Y = Xp - torch.conj(G.transpose(-2, -1)) @ X_delay
    return Y.permute(0, 2, 1, 3)


def projection_back(Y, ref):
    num = torch.sum(torch.conj(ref[:, :, :, None]) * Y, dim=1)
    denom = torch.sum(torch.abs(Y) ** 2, dim=1)
    c = torch.ones(num.shape, dtype=Y.dtype, device=Y.device)
    valid = denom > 0.0
    c[valid] = num[valid] / denom[valid]
    return c


def auxiva(X, n_src=None, n_iter=10, proj_back=True, W0=None, model="laplace"):
    n_batches, n_frames, n_freq, n_chan = X.shape
    device = X.device
    if n_src is None:
        n_src = n_chan
    W_hat = torch.zeros((n_batches, n_freq, n_chan, n_chan), dtype=X.dtype, device=device)
    W = W_hat[:, :, :n_src, :]
    if W0 is None:
        W[:, :, :, :n_src] = torch.tile(torch.eye(n_src, n_src, dtype=X.dtype), (n_batches, n_freq, 1, 1)).to(device)
    else:
        W[:, :, :, :] = W0
    eps = 1e-10
    eyes = torch.tile(torch.eye(n_chan, n_chan, dtype=X.dtype), (n_batches, n_freq, 1, 1)).to(device)
    r_inv = torch.zeros((n_batches, n_src, n_frames), device=device)
    r = torch.zeros((n_batches, n_src, n_frames), device=device)
    Y = torch.zeros((n_batches, n_freq, n_src, n_frames), dtype=X.dtype, device=device)
    X_original = X
    X = X.permute(0, 2, 3, 1).clone()

    def demix(Y, X, W):
        Y[:, :, :, :] = torch.matmul(W, X)

    for _ in range(n_iter):
        demix(Y, X, W)
        if model == "laplace":
            r[:, :, :] = 2.0 * torch.norm(Y, dim=1)
        else:
            r[:, :, :] = (torch.norm(Y, dim=1) ** 2) / n_freq
        r[r < eps] = eps
        r_inv[:, :, :] = 1.0 / r
        for s in range(n_src):
            V = torch.matmul((X * r_inv[:, None, s, None, :]), torch.conj(X.swapaxes(2, 3))) / n_frames
            WV = torch.matmul(W_hat, V)
            W[:, :, s, :] = torch.conj(torch.linalg.solve(WV + eps * eyes, eyes[:, :, :, s]))
            denom = torch.matmul(torch.matmul(W[:, :, None, s, :], V[:, :, :, :]), torch.conj(W[:, :, s, :, None]))
            W[:, :, s, :] /= torch.sqrt(denom[:, :, :, 0] + eps * torch.ones((n_batches, n_freq, 1), device=device))

    demix(Y, X, W)
    Y = Y.permute(0, 3, 1, 2).clone()
    if proj_back:
        z = projection_back(Y, X_original[:, :, :, 0])
        Y *= torch.conj(z[:, None, :, :])
    return Y


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced wav from noisy wav using the official H-GTCRN CPU chain.")
    parser.add_argument("--model", default="../models/model.axmodel")
    parser.add_argument("--input-wav", default="../samples/Samples1_noisy.wav")
    parser.add_argument("--output", default="../samples/Samples1_board_enhanced.wav")
    args = parser.parse_args()

    noisy, sr = sf.read(args.input_wav, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16000 Hz input, got {sr}")
    if noisy.ndim == 1:
        noisy = np.stack([noisy, noisy], axis=1)
    x = torch.from_numpy(noisy.T).unsqueeze(0)

    n_fft = 512
    hop_len = 256
    win_len = 512
    stft_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_len,
        "win_length": win_len,
        "window": torch.hann_window(win_len, device=x.device),
        "onesided": True,
    }
    spec_orig = multi_channel_stft(x.transpose(1, 2), **stft_kwargs)
    spec_drb = fd_wpe(spec_orig, rt60=0.3, shift=256, D=2, fs=16000, num_iter=1)
    spec_2ch = auxiva(spec_drb.transpose(1, 3), n_iter=10).transpose(1, 3)

    spec_norm = torch.norm(spec_2ch, dim=(2, 3))
    pred = torch.where(spec_norm[:, 0] < spec_norm[:, 1], 1, 0).view(-1, 1, 1, 1)
    spec_selected = spec_2ch[:, 0] * pred[:, 0] + spec_2ch[:, 1] * (1 - pred[:, 0])
    spec_unselected = spec_2ch[:, 1] * pred[:, 0] + spec_2ch[:, 0] * (1 - pred[:, 0])

    spec_sel = torch.view_as_real(spec_selected).permute(0, 3, 2, 1)
    spec_sel_log = torch.log10(torch.norm(spec_sel, dim=1, keepdim=True).clamp(1e-12))
    spec_un = torch.view_as_real(spec_unselected).permute(0, 3, 2, 1)
    spec_un_log = torch.log10(torch.norm(spec_un, dim=1, keepdim=True).clamp(1e-12))
    spec = torch.view_as_real(spec_orig)
    spec = spec.permute(0, 1, 4, 3, 2).reshape(spec.shape[0], -1, spec.shape[3], spec.shape[2])
    feat = torch.cat([spec, spec_sel_log, spec_un_log], dim=1).contiguous().cpu().numpy().astype(np.float32)

    session = ModelSession(args.model)
    mask = session.run_named({"feat": feat})["mask"]

    spec = torch.view_as_real(spec_orig)
    spec = spec.permute(0, 1, 4, 3, 2).reshape(spec.shape[0], -1, spec.shape[3], spec.shape[2])
    m = torch.from_numpy(mask).to(x.device)
    s_real = spec[:, 0] * m[:, 0] - spec[:, 1] * m[:, 1]
    s_imag = spec[:, 1] * m[:, 0] + spec[:, 0] * m[:, 1]
    spec_enh = torch.stack([s_real, s_imag], dim=1)
    spec_enh = spec_enh.permute(0, 3, 2, 1)
    spec_enh = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
    out = torch.istft(spec_enh, **stft_kwargs)
    out = torch.nn.functional.pad(out, (0, x.shape[-1] - out.shape[-1]))
    sf.write(args.output, out.cpu().numpy().squeeze(), sr)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
