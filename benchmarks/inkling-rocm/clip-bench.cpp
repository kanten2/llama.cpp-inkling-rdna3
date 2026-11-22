// Compile against the same ggml-base and ggml libraries as the experimental HIP backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static void measure(ggml_backend_t backend, const char * name, int width, int rows, bool scaled, const char * prefix) {
    constexpr int copies = 32;
    constexpr int replays = 64;
    auto * ctx = ggml_init({8u << 20, nullptr, true});
    auto * graph = ggml_new_graph(ctx);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    auto * weight = scaled ? ggml_dup_tensor(ctx, x) : nullptr;
    const float threshold = width == 16 ? 8.0f : std::sqrt(512.0f);
    ggml_tensor * out = nullptr;
    for (int i = 0; i < copies; ++i) {
        auto * v = scaled ? ggml_mul(ctx, x, weight) : x;
        auto * norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, v)));
        norm = ggml_scale_bias(ctx, norm, 1.0f, 1e-6f);
        auto * ratio = ggml_div(ctx, ggml_fill(ctx, norm, threshold), norm);
        out = ggml_mul(ctx, v, ggml_clamp(ctx, ratio, 0.0f, 1.0f));
        ggml_build_forward_expand(graph, out);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<float> values(width * rows), weights(width * rows);
    for (int i = 0; i < width * rows; ++i) {
        int row = i / width;
        values[i] = row == 0 ? 0 : row == 1 ? threshold / std::sqrt(float(width)) :
                std::sin(float(i)) * (row % 2 ? 0.25f : 9.0f);
        weights[i] = row == 1 ? 1.0f : 0.7f + 0.4f * std::cos(float(i));
    }
    ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
    if (weight) {
        ggml_backend_tensor_set(weight, weights.data(), 0, ggml_nbytes(weight));
    }
    auto run = [&] { GGML_ASSERT(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS); };
    for (int i = 0; i < 12; ++i) { run(); }
    ggml_backend_synchronize(backend);
    std::vector<double> times;
    for (int batch = 0; batch < 9; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < replays; ++i) { run(); }
        ggml_backend_synchronize(backend);
        const auto duration = std::chrono::steady_clock::now() - start;
        times.push_back(std::chrono::duration<double, std::micro>(duration).count() / (copies * replays));
    }
    ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
    std::ofstream dump(std::string(prefix) + "-" + name + ".f32", std::ios::binary);
    dump.write(reinterpret_cast<const char *>(values.data()), ggml_nbytes(out));
    GGML_ASSERT(dump.good());
    std::sort(times.begin(), times.end());
    std::printf("%s %d %d median_us %.5f\n", name, width, rows, times[times.size()/2]);

    // Vary every row: a fixed input can miss FP32 differences in the sum of squares.
    std::mt19937 random(123);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::ofstream stress(std::string(prefix) + "-" + name + "-random.f32", std::ios::binary);
    for (int sample = 0; sample < 256; ++sample) {
        for (int i = 0; i < width * rows; ++i) {
            const int row = i / width;
            values[i] = normal(random) * (threshold / std::sqrt(float(width))) * (0.75f + 0.3f * (row % 9));
            weights[i] = normal(random);
        }
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        if (weight) {
            ggml_backend_tensor_set(weight, weights.data(), 0, ggml_nbytes(weight));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
        stress.write(reinterpret_cast<const char *>(values.data()), ggml_nbytes(out));
    }
    GGML_ASSERT(stress.good());
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: clip-bench HIP-backend-DLL output-prefix\n");
        return 2;
    }
    GGML_ASSERT(ggml_backend_load(argv[1]));
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    GGML_ASSERT(dev);
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    std::printf("device %s\n", ggml_backend_dev_description(dev));
    measure(backend, "Q", 128, 24, true, argv[2]);
    measure(backend, "K", 128, 8, false, argv[2]);
    measure(backend, "R", 16, 24, true, argv[2]);
    ggml_backend_free(backend);
}
