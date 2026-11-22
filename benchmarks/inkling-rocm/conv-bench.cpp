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

static void build_history(ggml_context * ctx, ggml_cgraph * graph, int channels,
                          std::vector<ggml_tensor *> & inputs, std::vector<ggml_tensor *> & outputs) {
    auto * past = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, channels, 1);
    auto * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 1);
    constexpr int guard = 8;
    auto * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*channels + 2*guard);
    inputs.insert(inputs.end(), {past, cur, state});
    auto * window = ggml_concat(ctx, past, ggml_transpose(ctx, cur), 0);
    auto * tail = ggml_view_3d(ctx, window, 3, channels, 1, window->nb[1], window->nb[2], sizeof(float));
    auto * cell = ggml_view_2d(ctx, state, 3*channels, 1, state->nb[1], guard*sizeof(float));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, tail, cell));
    outputs.insert(outputs.end(), {window, state});
}

static void measure(ggml_backend_t gpu, int channels, const char * prefix, bool dump, bool history) {
    constexpr int copies = 32;
    ggml_context * ctx = ggml_init({8u << 20, nullptr, true});
    auto * graph = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> inputs, outputs;
    for (int i = 0; i < copies; ++i) {
        if (history) {
            build_history(ctx, graph, channels, inputs, outputs);
            continue;
        }
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, channels, 1);
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
        auto * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 1);
        inputs.insert(inputs.end(), {x, w, residual});
        auto * conv = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, w), channels, 1);
        auto * out = ggml_scale(ctx, ggml_add(ctx, residual, conv), 1.0f/std::sqrt(5.0f));
        ggml_build_forward_expand(graph, out);
        outputs.push_back(out);
    }
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    GGML_ASSERT(buf);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto fill = [&] {
        for (auto * t : inputs) {
            std::vector<float> v(ggml_nelements(t));
            for (float & value : v) { value = dist(gen); }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    };
    auto run = [&] { GGML_ASSERT(ggml_backend_graph_compute_async(gpu, graph) == GGML_STATUS_SUCCESS); };
    fill();
    for (int i = 0; i < 10; ++i) { run(); }
    ggml_backend_synchronize(gpu);
    std::vector<double> times;
    constexpr int replays = 64;
    for (int batch = 0; batch < 9; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < replays; ++i) { run(); }
        ggml_backend_synchronize(gpu);
        const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        times.push_back(us/(copies*replays));
    }
    std::sort(times.begin(), times.end());
    std::printf("channels=%d us_per_%s=%.4f\n", channels, history ? "history" : "conv", times[times.size()/2]);
    if (dump) {
        std::ofstream file(std::string(prefix) + "-" + std::to_string(channels) + ".f32", std::ios::binary);
        for (int i = 0; i < 16; ++i) {
            fill();
            run();
            ggml_backend_synchronize(gpu);
            for (auto * t : outputs) {
                std::vector<float> v(ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
                file.write(reinterpret_cast<const char *>(v.data()), ggml_nbytes(t));
            }
        }
        GGML_ASSERT(file.good());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const bool history = argc > 1 && std::string(argv[1]) == "--history";
    if (history) { --argc; ++argv; }
    if (argc != 3 && argc != 4) { return 2; }
    GGML_ASSERT(ggml_backend_load(argv[1]));
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    GGML_ASSERT(dev);
    auto * gpu = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(gpu);
    std::printf("device %s\n", ggml_backend_dev_description(dev));
    for (int channels : {1024, 2304}) {
        measure(gpu, channels, argv[2], argc == 4, history);
    }
    ggml_backend_free(gpu);
}
