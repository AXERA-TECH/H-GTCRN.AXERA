#include "model_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/audio_chain.hpp"

namespace {

std::vector<float> read_float_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    std::vector<char> bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()};
    std::vector<float> data(bytes.size() / sizeof(float));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

void write_float_file(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// feat.bin → mask.bin（原 core-only 模式）
int run_core(const std::string& model_path, const std::vector<std::string>& inputs,
             const std::string& output_dir, int bench, double audio_seconds) {
    std::vector<std::vector<float>> in;
    for (const auto& p : inputs) in.push_back(read_float_file(p));
    ModelRunner runner(model_path, "model");
    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> outputs = runner.Run(in);
    auto stop = std::chrono::steady_clock::now();
    const double first_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    double bench_ms = first_ms;
    if (bench > 1) {
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) {
            outputs = runner.Run(in);
        }
        stop = std::chrono::steady_clock::now();
        bench_ms = std::chrono::duration<double, std::milli>(stop - start).count() / bench;
    }
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    std::system(mkdir_cmd.c_str());
    for (size_t i = 0; i < outputs.size(); ++i) {
        write_float_file(
            output_dir + "/output_" + std::to_string(i) + ".bin", outputs[i]);
    }
    std::printf("inputs=%zu outputs=%zu\n", runner.NumInputs(), runner.NumOutputs());
    std::printf("first_run_ms: %.3f\n", first_ms);
    std::printf("bench_avg_ms: %.3f\n", bench_ms);
    std::printf("core_rtf: %.6f\n", bench_ms / 1000.0 / audio_seconds);
    return 0;
}

// wav → wav 端到端模式（CPU: STFT/WPE/IVA/ISTFT，NPU: GTCRN 核）
int run_wav(const std::string& model_path, const std::string& input_wav,
            const std::string& output_wav, int bench, double audio_seconds) {
    int ch = 0, sr = 0;
    std::vector<double> wav_flat;
    if (!hg::read_wav(input_wav, ch, sr, wav_flat)) return 1;
    if (sr != hg::kSR) {
        std::fprintf(stderr, "hg: expected %d Hz, got %d\n", hg::kSR, sr);
        return 1;
    }
    int L = (int)wav_flat.size() / ch;
    std::vector<std::vector<double>> x(ch, std::vector<double>(L));
    for (int c = 0; c < ch; c++)
        for (int t = 0; t < L; t++) x[c][t] = wav_flat[(size_t)c * L + t];

    // CPU 链路 → feat
    std::vector<float> feat;
    std::vector<hg::Spec> spec_orig;
    auto t0 = std::chrono::steady_clock::now();
    if (!hg::build_feat(x, feat, spec_orig)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // NPU 核 → mask
    ModelRunner runner(model_path, "model");
    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> outputs = runner.Run({feat});
    auto stop = std::chrono::steady_clock::now();
    const double first_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    double bench_ms = first_ms;
    if (bench > 1) {
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) outputs = runner.Run({feat});
        stop = std::chrono::steady_clock::now();
        bench_ms = std::chrono::duration<double, std::milli>(stop - start).count() / bench;
    }

    // 掩码应用 + ISTFT（mask[2][T][F]，spec 平面 0/1 = ch0 re/im）
    const std::vector<float>& m = outputs[0];
    int T = (int)spec_orig[0][0].size();
    hg::Spec spec_enh(hg::kFreq, std::vector<hg::cpx>(T));
    for (int t = 0; t < T; t++)
        for (int f = 0; f < hg::kFreq; f++) {
            float re = m[((size_t)0 * hg::kMaxFrames + t) * hg::kFreq + f];
            float im = m[((size_t)1 * hg::kMaxFrames + t) * hg::kFreq + f];
            hg::cpx s0 = spec_orig[0][f][t];
            spec_enh[f][t] = hg::cpx(s0.real() * re - s0.imag() * im,
                                     s0.imag() * re + s0.real() * im);
        }
    std::vector<double> out = hg::istft(spec_enh);
    out.resize(L);

    if (!hg::write_wav(output_wav, out, sr)) return 1;

    std::printf("cpu_chain_ms: %.1f\n", cpu_ms);
    std::printf("first_run_ms: %.3f\n", first_ms);
    std::printf("bench_avg_ms: %.3f\n", bench_ms);
    std::printf("core_rtf: %.6f\n", bench_ms / 1000.0 / audio_seconds);
    std::printf("saved: %s\n", output_wav.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // 用法1（wav→wav 端到端）:
    //   h_gtcrn_ax650 <model.axmodel> <in.wav> <out.wav> [--bench N] [--audio-seconds S]
    // 用法2（core-only，原接口）:
    //   h_gtcrn_ax650 <model.axmodel> <input_0.bin> [...] <output_dir> [--bench N] [--audio-seconds S]
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <model.axmodel> <in.wav> <out.wav> [--bench N] [--audio-seconds S]\n"
                     "   or: %s <model.axmodel> <input_0.bin> [...] <output_dir> [--bench N] [--audio-seconds S]\n",
                     argv[0], argv[0]);
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        int bench = 1;
        double audio_seconds = 10.0;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
                bench = std::atoi(argv[i + 1]);
            } else if (std::strcmp(argv[i], "--audio-seconds") == 0 && i + 1 < argc) {
                audio_seconds = std::atof(argv[i + 1]);
            }
        }
        if (ends_with(argv[2], ".wav")) {
            return run_wav(model_path, argv[2], argv[3], bench, audio_seconds);
        }
        int positional_end = argc;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--bench") == 0) positional_end = std::min(positional_end, i);
            else if (std::strcmp(argv[i], "--audio-seconds") == 0) positional_end = std::min(positional_end, i);
        }
        std::vector<std::string> inputs;
        for (int i = 2; i < positional_end - 1; ++i) inputs.push_back(argv[i]);
        return run_core(model_path, inputs, argv[positional_end - 1], bench, audio_seconds);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
