#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static void check_projection(ggml_backend_t gpu, gguf_context * model, ggml_context * metadata,
                             std::ifstream & weights, const char * name, const char * prefix, bool gate, bool time_graph) {
    const int64_t index = gguf_find_tensor(model, name);
    GGML_ASSERT(index >= 0);
    const auto * desc = ggml_get_tensor(metadata, name);
    GGML_ASSERT(desc && desc->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(desc->ne[0] == (gate ? 2304 : 1152) && desc->ne[1] == 2304 && desc->ne[2] == 128);

    auto * ctx = ggml_init({4u << 20, nullptr, true});
    auto * w = ggml_dup_tensor(ctx, desc);
    std::array<ggml_tensor *, 16> xs, selections, outputs;
    auto * graph = ggml_new_graph(ctx);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, desc->ne[0], gate ? 1 : 8);
        selections[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
        outputs[i] = ggml_mul_mat_id(ctx, w, xs[i], selections[i]);
        ggml_build_forward_expand(graph, outputs[i]);
    }
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    GGML_ASSERT(buf);
    weights.seekg(gguf_get_data_offset(model) + gguf_get_tensor_offset(model, index));
    std::vector<char> bank(w->nb[2]);
    for (int64_t expert = 0; expert < w->ne[2]; ++expert) {
        weights.read(bank.data(), bank.size());
        GGML_ASSERT(weights.good());
        ggml_backend_tensor_set(w, bank.data(), expert*w->nb[2], bank.size());
    }

    std::ofstream file(std::string(prefix) + (gate ? "-gate.f32" : "-down.f32"), std::ios::binary);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist;
    std::vector<float> input(ggml_nelements(xs.front())), result(ggml_nelements(outputs.front()));
    for (size_t trial = 0; trial < xs.size(); ++trial) {
        for (auto & v : input) { v = dist(gen); }
        std::array<int32_t, 8> chosen;
        for (size_t i = 0; i < chosen.size(); ++i) { chosen[i] = ((8*trial + i)*37) % w->ne[2]; }
        ggml_backend_tensor_set(xs[trial], input.data(), 0, ggml_nbytes(xs[trial]));
        ggml_backend_tensor_set(selections[trial], chosen.data(), 0, ggml_nbytes(selections[trial]));
    }
    GGML_ASSERT(ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS);
    for (auto * out : outputs) {
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        file.write(reinterpret_cast<const char *>(result.data()), ggml_nbytes(out));
    }
    file.close();
    GGML_ASSERT(file.good());
    if (time_graph) {
        auto replay = [&] { GGML_ASSERT(ggml_backend_graph_compute_async(gpu, graph) == GGML_STATUS_SUCCESS); };
        for (int i = 0; i < 12; ++i) { replay(); }
        ggml_backend_synchronize(gpu);
        std::array<double, 9> samples;
        for (auto & us : samples) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 32; ++i) { replay(); }
            ggml_backend_synchronize(gpu);
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/(32*xs.size());
        }
        std::sort(samples.begin(), samples.end());
        std::printf("%s median_us=%.4f\n", name, samples[samples.size()/2]);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--time")) {
        std::fprintf(stderr, "usage: iq4-gate-check <backend.dll> <model.gguf> <output-prefix> [--time]\n");
        return 2;
    }
    GGML_ASSERT(ggml_backend_load(argv[1]));
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    GGML_ASSERT(dev);
    auto * gpu = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(gpu);
    ggml_context * metadata = nullptr;
    auto * model = gguf_init_from_file(argv[2], {true, &metadata});
    GGML_ASSERT(model && metadata);
    std::ifstream weights(argv[2], std::ios::binary);
    check_projection(gpu, model, metadata, weights, "blk.0.ffn_gate_up_exps.weight", argv[3], true, argc == 5);
    check_projection(gpu, model, metadata, weights, "blk.0.ffn_down_exps.weight", argv[3], false, argc == 5);
    ggml_free(metadata);
    gguf_free(model);
    ggml_backend_free(gpu);
}
