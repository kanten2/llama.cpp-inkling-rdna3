#include "../../src/models/inkling2-mix.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>

static constexpr int width = 2304;
static constexpr int copies = 12;

static void measure(ggml_backend_t gpu, const std::string & mode, int states, int queries,
                    const char * path, bool dump) {
    ggml_context * ctx = ggml_init({8u << 20, nullptr, true});
    auto * graph = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> inputs, outputs, blocks;
    auto input = [&](int cols) {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, cols);
        inputs.push_back(t);
        return t;
    };
    for (int i = 1; i < states; ++i) {
        blocks.push_back(input(1));
    }
    auto * bank_cur = input(1);
    inkling2_decode_mix cache;
    for (int i = 0; i < copies; ++i) {
        auto * cur = queries == 4 ? bank_cur : input(1);
        auto * q = input(queries);
        ggml_tensor * out;
        if (mode == "transpose") {
            out = ggml_cont(ctx, ggml_transpose(ctx, input(states)));
        } else if (mode == "reuse") {
            out = cache.build(ctx, blocks, cur, q, queries == 4);
        } else {
            out = ggml_reshape_3d(ctx, cur, width, 1, 1);
            for (auto * block : blocks) {
                out = ggml_concat(ctx, ggml_reshape_3d(ctx, block, width, 1, 1), out, 1);
            }
            if (mode == "stock") {
                auto * norm = ggml_rms_norm(ctx, out, 1e-6f);
                auto * repeated = ggml_repeat(ctx, q, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, queries, 1));
                auto * scores = ggml_mul_mat(ctx, norm, repeated);
                scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/sqrtf(width), 0.0f);
                out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, norm)), scores);
            }
        }
        ggml_build_forward_expand(graph, out);
        outputs.push_back(out);
    }
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    GGML_ASSERT(buf);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0, 1);
    auto fill = [&] {
        for (auto * t : inputs) {
            std::vector<float> v(ggml_nelements(t));
            for (auto & value : v) { value = dist(gen); }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    };
    fill();
    auto run = [&] { GGML_ASSERT(ggml_backend_graph_compute_async(gpu, graph) == GGML_STATUS_SUCCESS); };
    for (int i = 0; i < 10; ++i) { run(); }
    ggml_backend_synchronize(gpu);
    std::vector<double> times;
    constexpr int replays = 32;
    for (int batch = 0; batch < 7; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < replays; ++i) { run(); }
        ggml_backend_synchronize(gpu);
        const auto us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        times.push_back(us/(copies*replays));
    }
    std::sort(times.begin(),times.end());
    std::printf("%s %d %d %.4f\n",mode.c_str(),states,queries,times[times.size()/2]);
    if (dump) {
        std::ofstream file(std::string(path)+"-"+std::to_string(states)+"-"+std::to_string(queries)+".f32",std::ios::binary);
        for (int seed = 0; seed < 16; ++seed) {
            fill();
            run();
            ggml_backend_synchronize(gpu);
            for (auto * t : outputs) {
                std::vector<float> v(ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
                file.write(reinterpret_cast<const char *>(v.data()),ggml_nbytes(t));
            }
        }
        GGML_ASSERT(file.good());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) { return 2; }
    const std::string mode = argv[2];
    if (mode != "stock" && mode != "reuse" && mode != "concat" && mode != "transpose") { return 2; }
    GGML_ASSERT(ggml_backend_load(argv[1]));
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    GGML_ASSERT(dev);
    auto * gpu = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(gpu);
    std::printf("device %s\n", ggml_backend_dev_description(dev));
    for (int s = 2; s <= 5; ++s) {
        for (int q : {1,3,4}) {
            if (q != 4 || s == 2) { measure(gpu,mode,s,q,argv[3],argc == 5); }
        }
    }
    ggml_backend_free(gpu);
}
