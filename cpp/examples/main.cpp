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

}  // namespace

int main(int argc, char** argv) {
    // usage: model_example <model.axmodel> <input_0.bin> [input_1.bin ...] <output_dir> [--bench N] [--audio-seconds S]
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: %s <model.axmodel> <input_0.bin> [input_1.bin ...] <output_dir> [--bench N] [--audio-seconds S]\n",
            argv[0]);
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        int bench = 1;
        double audio_seconds = 10.0;
        int positional_end = argc;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
                bench = std::atoi(argv[i + 1]);
                positional_end = std::min(positional_end, i);
                ++i;
            } else if (std::strcmp(argv[i], "--audio-seconds") == 0 && i + 1 < argc) {
                audio_seconds = std::atof(argv[i + 1]);
                positional_end = std::min(positional_end, i);
                ++i;
            }
        }
        const std::string output_dir = argv[positional_end - 1];
        std::string mkdir_cmd = "mkdir -p " + output_dir;
        std::system(mkdir_cmd.c_str());
        std::vector<std::vector<float>> inputs;
        for (int i = 2; i < positional_end - 1; ++i) {
            inputs.push_back(read_float_file(argv[i]));
        }
        ModelRunner runner(model_path, "model");
        auto start = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> outputs = runner.Run(inputs);
        auto stop = std::chrono::steady_clock::now();
        const double first_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        double bench_ms = first_ms;
        if (bench > 1) {
            start = std::chrono::steady_clock::now();
            for (int i = 0; i < bench; ++i) {
                outputs = runner.Run(inputs);
            }
            stop = std::chrono::steady_clock::now();
            bench_ms = std::chrono::duration<double, std::milli>(stop - start).count() / bench;
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            write_float_file(
                output_dir + "/output_" + std::to_string(i) + ".bin", outputs[i]);
        }
        std::printf("inputs=%zu outputs=%zu\n", runner.NumInputs(), runner.NumOutputs());
        std::printf("first_run_ms: %.3f\n", first_ms);
        std::printf("bench_avg_ms: %.3f\n", bench_ms);
        std::printf("core_rtf: %.6f\n", bench_ms / 1000.0 / audio_seconds);
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
