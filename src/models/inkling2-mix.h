#pragma once

#include "ggml.h"
#include <cmath>
#include <vector>

// Graph-local reuse of immutable inputs during a single-token forward pass.
struct inkling2_decode_mix {
    std::vector<ggml_tensor *> prefix_blocks;
    ggml_tensor * prefix = nullptr;
    std::vector<ggml_tensor *> bank_blocks;
    ggml_tensor * bank_cur = nullptr;
    ggml_tensor * bank_states = nullptr;
    ggml_tensor * bank_transposed = nullptr;

    ggml_tensor * build(ggml_context * ctx, const std::vector<ggml_tensor *> & blocks,
                        ggml_tensor * cur, ggml_tensor * query, bool message_bank) {
        const int64_t width = cur->ne[0];
        ggml_tensor * states;
        ggml_tensor * transposed;
        if (message_bank && cur == bank_cur && blocks == bank_blocks) {
            states = bank_states;
            transposed = bank_transposed;
        } else {
            states = ggml_reshape_3d(ctx, cur, width, 1, 1);
            if (!blocks.empty()) {
                // The stock graph prepends each block, in reverse storage order.
                ggml_tensor * stack = prefix;
                if (message_bank || blocks != prefix_blocks) {
                    stack = ggml_reshape_3d(ctx, blocks.front(), width, 1, 1);
                    for (size_t j = 1; j < blocks.size(); ++j) {
                        stack = ggml_concat(ctx, ggml_reshape_3d(ctx, blocks[j], width, 1, 1), stack, 1);
                    }
                    if (!message_bank) {
                        prefix_blocks = blocks;
                        prefix = stack;
                    }
                }
                states = ggml_concat(ctx, stack, states, 1);
            }
            states = ggml_rms_norm(ctx, states, 1e-6f);
            transposed = ggml_cont(ctx, ggml_transpose(ctx, states));
            if (message_bank) {
                bank_blocks = blocks;
                bank_cur = cur;
                bank_states = states;
                bank_transposed = transposed;
            }
        }
        auto * scores = ggml_mul_mat(ctx, states, query);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/sqrtf(width), 0.0f);
        return ggml_mul_mat(ctx, transposed, scores);
    }
};
