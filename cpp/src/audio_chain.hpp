// H-GTCRN 端到端 CPU 链路：STFT → WPE → auxIVA → 特征构造 → [NPU 核] → 掩码应用 → ISTFT
// 与 python/audio_demo.py（numpy 版）逐式对齐（官方 GTCRN_IVA CPU 链路）。
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace hg {

constexpr int kNfft = 512;
constexpr int kHop = 256;
constexpr int kFreq = kNfft / 2 + 1;  // 257
constexpr int kMaxFrames = 626;       // feat[1,6,626,257]，约 10.0s @16kHz
constexpr int kSR = 16000;

using cpx = std::complex<double>;

inline std::vector<double> hann_window(int n) {  // periodic hann（等价 torch.hann_window）
    std::vector<double> w(n);
    for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n);
    return w;
}

// ---------------- 复数小矩阵求逆（高斯消元，部分主元） ----------------
inline std::vector<cpx> mat_inv(const std::vector<cpx>& a, int n) {
    std::vector<cpx> aug((size_t)n * 2 * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            aug[(size_t)i * 2 * n + j] = a[(size_t)i * n + j];
            aug[(size_t)i * 2 * n + n + j] = (i == j) ? cpx(1.0, 0.0) : cpx(0.0, 0.0);
        }
    for (int col = 0; col < n; col++) {
        int piv = col;
        double best = std::abs(aug[(size_t)col * 2 * n + col]);
        for (int r = col + 1; r < n; r++) {
            double v = std::abs(aug[(size_t)r * 2 * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-30) { std::fprintf(stderr, "hg: singular matrix in mat_inv\n"); break; }
        if (piv != col)
            for (int j = 0; j < 2 * n; j++)
                std::swap(aug[(size_t)col * 2 * n + j], aug[(size_t)piv * 2 * n + j]);
        cpx d = aug[(size_t)col * 2 * n + col];
        for (int j = 0; j < 2 * n; j++) aug[(size_t)col * 2 * n + j] /= d;
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            cpx f = aug[(size_t)r * 2 * n + col];
            if (std::abs(f) < 1e-30) continue;
            for (int j = 0; j < 2 * n; j++) aug[(size_t)r * 2 * n + j] -= f * aug[(size_t)col * 2 * n + j];
        }
    }
    std::vector<cpx> out((size_t)n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) out[(size_t)i * n + j] = aug[(size_t)i * 2 * n + n + j];
    return out;
}

// ---------------- FFT（radix-2 迭代） ----------------
inline void fft_inplace(std::vector<cpx>& a, bool inverse) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = (inverse ? 2.0 : -2.0) * M_PI / len;
        cpx wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cpx w(1.0, 0.0);
            for (int k = 0; k < len / 2; k++) {
                cpx u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse)
        for (auto& x : a) x /= (double)n;
}

inline std::vector<cpx> rfft(const std::vector<double>& x, int n) {
    std::vector<cpx> a(n);
    for (int i = 0; i < n; i++) a[i] = cpx(x[i], 0.0);
    fft_inplace(a, false);
    a.resize(n / 2 + 1);
    return a;
}

inline void irfft(const std::vector<cpx>& X, int n, std::vector<double>& out) {
    std::vector<cpx> a(n);
    for (int i = 0; i < n / 2 + 1; i++) a[i] = X[i];
    for (int i = n / 2 + 1; i < n; i++) a[i] = std::conj(X[n - i]);
    fft_inplace(a, true);
    out.resize(n);
    for (int i = 0; i < n; i++) out[i] = a[i].real();
}

// ---------------- WAV 读写（PCM16） ----------------
inline bool read_wav(const std::string& path, int& ch, int& sr, std::vector<double>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "hg: cannot open %s\n", path.c_str()); return false; }
    char hdr[44];
    f.read(hdr, 44);
    if (std::memcmp(hdr, "RIFF", 4) || std::memcmp(hdr + 8, "WAVE", 4)) {
        std::fprintf(stderr, "hg: %s is not a wav file\n", path.c_str());
        return false;
    }
    int16_t ch16; std::memcpy(&ch16, hdr + 22, 2);
    int32_t sr32; std::memcpy(&sr32, hdr + 24, 4);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    int n = (int)raw.size() / 2;
    const int16_t* p = reinterpret_cast<const int16_t*>(raw.data());
    int L = n / (ch16 > 0 ? ch16 : 1);
    int inch = (ch16 == 1) ? 1 : 2;
    data.assign((size_t)inch * L, 0.0);
    for (int t = 0; t < L; t++)
        for (int c = 0; c < inch; c++) {
            double v = p[(size_t)t * ch16 + c] / 32768.0;
            data[(size_t)c * L + t] = v;
            if (ch16 == 1) data[(size_t)L + t] = v;
        }
    ch = inch;
    sr = sr32;
    return true;
}

inline bool write_wav(const std::string& path, const std::vector<double>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "hg: cannot write %s\n", path.c_str()); return false; }
    int32_t n = (int32_t)x.size(), byte_rate = sr * 2, data_bytes = n * 2, chunk = 36 + data_bytes;
    int16_t one = 1, two = 2, sixteen = 16;
    f.write("RIFF", 4); f.write(reinterpret_cast<const char*>(&chunk), 4); f.write("WAVE", 4);
    f.write("fmt ", 4);
    int32_t sz = 16;
    f.write(reinterpret_cast<const char*>(&sz), 4);
    f.write(reinterpret_cast<const char*>(&one), 2);
    f.write(reinterpret_cast<const char*>(&one), 2);
    f.write(reinterpret_cast<const char*>(&sr), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&two), 2);
    f.write(reinterpret_cast<const char*>(&sixteen), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_bytes), 4);
    for (double v : x) {
        double c = std::max(-1.0, std::min(1.0, v));
        int16_t s = (int16_t)(c * 32767.0);
        f.write(reinterpret_cast<const char*>(&s), 2);
    }
    return true;
}

// ---------------- STFT / ISTFT ----------------
using Spec = std::vector<std::vector<cpx>>;  // [F][T]

// x[ch][L] → [ch][F][T]（center=True 反射填充 + periodic hann，与 numpy 版对齐）
inline std::vector<Spec> stft(const std::vector<std::vector<double>>& x) {
    int ch = (int)x.size(), L = (int)x[0].size();
    std::vector<double> win = hann_window(kNfft);
    int pad = kNfft / 2, padL = L + 2 * pad;
    std::vector<std::vector<double>> xp(ch, std::vector<double>(padL));
    for (int c = 0; c < ch; c++)
        for (int t = 0; t < padL; t++) {
            int s = t - pad;
            if (s < 0) s = -s;
            if (s >= L) s = 2 * L - 2 - s;
            xp[c][t] = x[c][std::max(0, std::min(L - 1, s))];
        }
    int T = (padL - kNfft) / kHop + 1;
    std::vector<Spec> spec(ch, Spec(kFreq, std::vector<cpx>(T)));
    std::vector<double> frame(kNfft);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < ch; c++) {
            for (int n = 0; n < kNfft; n++) frame[n] = xp[c][t * kHop + n] * win[n];
            std::vector<cpx> X = rfft(frame, kNfft);
            for (int f = 0; f < kFreq; f++) spec[c][f][t] = X[f];
        }
    return spec;
}

// spec[F][T] → (T-1)*hop + n_fft 长度信号（overlap-add + 窗能量归一），裁掉 center 填充
inline std::vector<double> istft(const Spec& spec) {
    int F = (int)spec.size();
    int T = (int)spec[0].size();
    std::vector<double> win = hann_window(kNfft);
    int out_len = (T - 1) * kHop + kNfft;
    std::vector<double> out(out_len, 0.0), wsum(out_len, 0.0);
    for (int t = 0; t < T; t++) {
        std::vector<cpx> Xt(F);
        for (int f = 0; f < F; f++) Xt[f] = spec[f][t];
        std::vector<double> frame;
        irfft(Xt, kNfft, frame);
        for (int n = 0; n < kNfft; n++) {
            out[t * kHop + n] += frame[n] * win[n];
            wsum[t * kHop + n] += win[n] * win[n];
        }
    }
    for (int i = 0; i < out_len; i++) out[i] /= (wsum[i] > 1e-10 ? wsum[i] : 1.0);
    int pad = kNfft / 2;
    return std::vector<double>(out.begin() + pad, out.end() - pad);
}

// ---------------- WPE 去混响 ----------------
// X[M][F][T] → [M][F][T]
inline std::vector<Spec> fd_wpe(const std::vector<Spec>& X, double rt60 = 0.3,
                                int shift = 256, int D = 2, int fs = 16000, int num_iter = 1) {
    int M = (int)X.size(), F = (int)X[0].size(), T = (int)X[0][0].size();
    double sum = 0.0;
    for (int f = 0; f < F; f++) {
        double mf = 0.0;
        for (int m = 0; m < M; m++) {
            double mt = 0.0;
            for (int t = 0; t < T; t++) mt = std::max(mt, std::norm(X[m][f][t]));
            mf = std::max(mf, mt);
        }
        sum += mf;
    }
    double eps = 1e-3 * sum / F;
    int Lg = (int)(rt60 * fs / shift);
    int Md = M * Lg;
    // Xp[F][M][T], Xd[F][Md][T]
    std::vector<std::vector<std::vector<cpx>>> Xp(F, std::vector<std::vector<cpx>>(M, std::vector<cpx>(T)));
    std::vector<std::vector<std::vector<cpx>>> Xd(F, std::vector<std::vector<cpx>>(Md, std::vector<cpx>(T)));
    for (int f = 0; f < F; f++) {
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) Xp[f][m][t] = X[m][f][t];
        for (int l = 0; l < Lg; l++)
            for (int m = 0; m < M; m++)
                for (int t = D + l; t < T; t++) Xd[f][l * M + m][t] = Xp[f][m][t - D - l];
        std::vector<std::vector<cpx>> Y(M, std::vector<cpx>(T));
        for (int m = 0; m < M; m++) Y[m] = Xp[f][m];
        for (int it = 0; it < num_iter; it++) {
            std::vector<double> lam(T);
            for (int t = 0; t < T; t++) {
                double s = 0.0;
                for (int m = 0; m < M; m++) s += std::norm(Y[m][t]);
                lam[t] = std::max(s / M, eps);
            }
            // temp[i][t] = Xd[i][t] / lam[t]（预计算，避免重复除法）
            std::vector<std::vector<cpx>> temp(Md, std::vector<cpx>(T));
            for (int i = 0; i < Md; i++)
                for (int t = 0; t < T; t++) temp[i][t] = Xd[f][i][t] / lam[t];
            // R 为 Hermitian：只算下三角，共轭对称填充
            std::vector<std::vector<cpx>> R(Md, std::vector<cpx>(Md)), P(Md, std::vector<cpx>(M));
            for (int i = 0; i < Md; i++)
                for (int j = i; j < Md; j++) {
                    cpx s(0, 0);
                    for (int t = 0; t < T; t++) s += temp[i][t] * std::conj(Xd[f][j][t]);
                    R[i][j] = s;
                    if (j != i) R[j][i] = std::conj(s);
                }
            for (int i = 0; i < Md; i++)
                for (int j = 0; j < M; j++) {
                    cpx s(0, 0);
                    for (int t = 0; t < T; t++) s += temp[i][t] * std::conj(Xp[f][j][t]);
                    P[i][j] = s;
                }
            std::vector<cpx> A((size_t)Md * Md);
            for (int i = 0; i < Md; i++)
                for (int j = 0; j < Md; j++)
                    A[(size_t)i * Md + j] = R[i][j] + (i == j ? eps : 0.0);
            std::vector<cpx> Rinv = mat_inv(A, Md);
            std::vector<std::vector<cpx>> G(Md, std::vector<cpx>(M));
            for (int i = 0; i < Md; i++)
                for (int j = 0; j < M; j++) {
                    cpx s(0, 0);
                    for (int k = 0; k < Md; k++) s += Rinv[(size_t)i * Md + k] * P[k][j];
                    G[i][j] = s;
                }
            for (int m = 0; m < M; m++)
                for (int t = 0; t < T; t++) {
                    cpx s(0, 0);
                    for (int i = 0; i < Md; i++) s += std::conj(G[i][m]) * Xd[f][i][t];
                    Y[m][t] = Xp[f][m][t] - s;
                }
        }
        for (int m = 0; m < M; m++) Xp[f][m] = Y[m];
    }
    std::vector<Spec> out(M, Spec(F, std::vector<cpx>(T)));
    for (int f = 0; f < F; f++)
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) out[m][f][t] = Xp[f][m][t];
    return out;
}

// ---------------- auxIVA 声源分离 ----------------
// X[T][F][M] → [T][F][M]（与 numpy 版 auxiva 对齐）
inline std::vector<std::vector<std::vector<cpx>>> auxiva(
    const std::vector<std::vector<std::vector<cpx>>>& X,
    int n_src = 2, int n_iter = 10, bool proj_back = true, const std::string& model = "laplace") {
    (void)model;  // 当前仅 laplace 模型（与 Python 版一致）
    int T = (int)X.size(), F = (int)X[0].size(), M = (int)X[0][0].size();
    const double eps = 1e-10;
    // W[F][M][M]，初始为单位阵（前 n_src 列）
    std::vector<std::vector<std::vector<cpx>>> W(F, std::vector<std::vector<cpx>>(M, std::vector<cpx>(M)));
    for (int f = 0; f < F; f++)
        for (int s = 0; s < n_src; s++) W[f][s][s] = cpx(1.0, 0.0);
    // Xp[F][M][T]
    std::vector<std::vector<std::vector<cpx>>> Xp(F, std::vector<std::vector<cpx>>(M, std::vector<cpx>(T)));
    for (int f = 0; f < F; f++)
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) Xp[f][m][t] = X[t][f][m];
    std::vector<std::vector<std::vector<cpx>>> Y(F, std::vector<std::vector<cpx>>(M, std::vector<cpx>(T)));
    std::vector<std::vector<double>> r(M, std::vector<double>(T));
    for (int it = 0; it < n_iter; it++) {
        for (int f = 0; f < F; f++)
            for (int m = 0; m < M; m++) {
                for (int t = 0; t < T; t++) {
                    cpx s(0, 0);
                    for (int k = 0; k < M; k++) s += W[f][m][k] * Xp[f][k][t];
                    Y[f][m][t] = s;
                }
            }
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) {
                double s = 0.0;
                for (int f = 0; f < F; f++) s += std::norm(Y[f][m][t]);
                r[m][t] = 2.0 * std::sqrt(s);  // laplace
            }
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) r[m][t] = std::max(r[m][t], eps);
        std::vector<std::vector<cpx>> V(M, std::vector<cpx>(M));
        for (int s = 0; s < n_src; s++) {
            for (int f = 0; f < F; f++) {
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < M; j++) {
                        cpx acc(0, 0);
                        for (int t = 0; t < T; t++)
                            acc += (Xp[f][i][t] / r[s][t]) * std::conj(Xp[f][j][t]);
                        V[i][j] = acc / (double)T;
                    }
                // WV = W @ V
                std::vector<std::vector<cpx>> WV(M, std::vector<cpx>(M));
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < M; j++) {
                        cpx acc(0, 0);
                        for (int k = 0; k < M; k++) acc += W[f][i][k] * V[k][j];
                        WV[i][j] = acc + (i == j ? eps : 0.0);
                    }
                // W[s,:] = conj(solve(WV + eps*I, e_s))
                std::vector<cpx> A((size_t)M * M);
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < M; j++) A[(size_t)i * M + j] = WV[i][j];
                std::vector<cpx> Ainv = mat_inv(A, M);
                for (int j = 0; j < M; j++) W[f][s][j] = std::conj(Ainv[(size_t)j * M + s]);
                // denom = W[s,:] @ V @ conj(W[s,:]); W[s,:] /= sqrt(denom + eps)
                cpx den(0, 0);
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < M; j++)
                        den += W[f][s][i] * V[i][j] * std::conj(W[f][s][j]);
                double norm = std::sqrt(std::max(den.real() + eps, 0.0));
                for (int j = 0; j < M; j++) W[f][s][j] /= norm;
            }
        }
    }
    // 最终 demix
    for (int f = 0; f < F; f++)
        for (int m = 0; m < M; m++)
            for (int t = 0; t < T; t++) {
                cpx s(0, 0);
                for (int k = 0; k < M; k++) s += W[f][m][k] * Xp[f][k][t];
                Y[f][m][t] = s;
            }
    // projection back：ref = X 的第 0 通道 [T][F]；对 T（帧）求和 → 每个 (f, 声源) 一个系数
    if (proj_back) {
        for (int f = 0; f < F; f++) {
            for (int m = 0; m < M; m++) {
                cpx n(0, 0);
                double d = 0.0;
                for (int t = 0; t < T; t++) {
                    n += std::conj(X[t][f][0]) * Y[f][m][t];
                    d += std::norm(Y[f][m][t]);
                }
                cpx c = (d > 0.0) ? n / d : cpx(1.0, 0.0);
                for (int t = 0; t < T; t++) Y[f][m][t] *= std::conj(c);
            }
        }
    }
    std::vector<std::vector<std::vector<cpx>>> out(T, std::vector<std::vector<cpx>>(F, std::vector<cpx>(M)));
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            for (int m = 0; m < M; m++) out[t][f][m] = Y[f][m][t];
    return out;
}

// ---------------- wav → feat + 增强（端到端） ----------------
// 输入 16kHz PCM16 wav（1/2 声道）；输出 feat[1,6,T,F]（T 补零到 kMaxFrames）与
// 原始 STFT 用于掩码应用；返回 false 表示不支持（长度/采样率）。
inline bool build_feat(const std::vector<std::vector<double>>& x,
                       std::vector<float>& feat, std::vector<Spec>& spec_orig) {
    int ch = (int)x.size(), L = (int)x[0].size();
    int T = L / kHop + 1;
    if (T > kMaxFrames) {
        std::fprintf(stderr, "hg: input too long: %d frames > %d (约 10.0s @16kHz)\n",
                     T, kMaxFrames);
        return false;
    }
    // 补零到 626 帧（零帧对 WPE/IVA 统计无贡献，proj_back 缩放不变性保证结果一致）
    std::vector<std::vector<double>> xp(ch, std::vector<double>(L));
    for (int c = 0; c < ch; c++)
        for (int t = 0; t < L; t++) xp[c][t] = x[c][t];
    if (T < kMaxFrames) {
        for (auto& v : xp) v.resize((kMaxFrames - 1) * kHop, 0.0);
    }
    spec_orig = stft(xp);  // [M][F][T]
    std::vector<Spec> spec_drb = fd_wpe(spec_orig);
    // → [T][F][M]
    std::vector<std::vector<std::vector<cpx>>> Xt(spec_drb[0][0].size(),
        std::vector<std::vector<cpx>>(kFreq, std::vector<cpx>(2)));
    for (int t = 0; t < (int)Xt.size(); t++)
        for (int f = 0; f < kFreq; f++)
            for (int m = 0; m < 2; m++) Xt[t][f][m] = spec_drb[m][f][t];
    auto sep = auxiva(Xt);  // [T][F][M]
    std::vector<Spec> spec_2ch(2, Spec(kFreq, std::vector<cpx>(kMaxFrames)));
    for (int t = 0; t < (int)sep.size(); t++)
        for (int f = 0; f < kFreq; f++)
            for (int m = 0; m < 2; m++) spec_2ch[m][f][t] = sep[t][f][m];
    // 通道选择：能量较小的通道
    double e0 = 0.0, e1 = 0.0;
    for (int f = 0; f < kFreq; f++)
        for (int t = 0; t < kMaxFrames; t++) {
            e0 += std::norm(spec_2ch[0][f][t]);
            e1 += std::norm(spec_2ch[1][f][t]);
        }
    int sel = (e0 < e1) ? 0 : 1, unsel = 1 - sel;
    // feat[6][T][F]：0-3 = ch0 re/im + ch1 re/im，4 = log10|sel|，5 = log10|unsel|
    feat.assign((size_t)6 * kMaxFrames * kFreq, 0.0f);
    auto put = [&](int plane, int t, int f, float v) {
        feat[((size_t)plane * kMaxFrames + t) * kFreq + f] = v;
    };
    for (int t = 0; t < kMaxFrames; t++)
        for (int f = 0; f < kFreq; f++) {
            put(0, t, f, (float)spec_orig[0][f][t].real());
            put(1, t, f, (float)spec_orig[0][f][t].imag());
            put(2, t, f, (float)spec_orig[1][f][t].real());
            put(3, t, f, (float)spec_orig[1][f][t].imag());
            put(4, t, f, (float)std::log10(std::max(std::abs(spec_2ch[sel][f][t]), 1e-12)));
            put(5, t, f, (float)std::log10(std::max(std::abs(spec_2ch[unsel][f][t]), 1e-12)));
        }
    // spec_orig 裁剪为实际帧数（掩码应用用原始 T，不用补零帧）
    for (auto& ff : spec_orig)
        for (auto& tt : ff) tt.resize(T);
    return true;
}

}  // namespace hg
