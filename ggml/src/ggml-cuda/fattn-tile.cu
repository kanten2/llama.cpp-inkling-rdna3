#include "common.cuh"
#include "fattn-tile.cuh"

#ifdef GGML_USE_HIP
#include "fattn-mma-f16.cuh"

struct inkling_head_bias {
    static constexpr bool has_far_tiles = false;
    inkling_relative_bias bias;
    int head;

    __device__ half read(int q, int k, int stride) const {
        return fattn_tile_bias(nullptr, stride, q, k, head, bias);
    }

    __device__ bool skip_tile(int, int) const { return false; }
};

static constexpr int inkling_tile_cols = 64;

struct inkling_global_head_bias : inkling_head_bias {
    static constexpr bool has_far_tiles = true;
    const uint8_t * far;
    int key_tiles;
    int last_row;
};

// A wave tests the actual row IDs for one query and one block of keys.
// The final (padded) table row already includes the per-query sink subtraction.
static __global__ void inkling_prepare_far_tiles(inkling_global_head_bias * heads, uint8_t * far,
                                                 inkling_relative_bias bias, int nk, int nq, int width, int last_row) {
    const int lane = threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (!blockIdx.x && !blockIdx.y && !threadIdx.y && lane < 24) {
        heads[lane] = {{bias, lane}, far, int(gridDim.x), last_row};
    }
    if (q >= nq) {
        return;
    }
    bool same = true;
    for (int i = lane; i < width && blockIdx.x * width + i < nk; i += 32) {
        same &= bias.rows[q * bias.index_stride + blockIdx.x * width + i] == last_row;
    }
    const bool all_far = __all(same);
    if (!lane) {
        far[q * gridDim.x + blockIdx.x] = all_far;
    }
}

struct inkling_swa_head_bias : inkling_head_bias {
    const uint8_t * visible;
    int key_tiles;

    __device__ bool skip_tile(int qt, int kt) const { return !visible[qt * key_tiles + kt]; }
};

// Test the actual mask so cache wraps, unused slots and partial tiles need no position assumptions.
static __global__ void inkling_prepare_swa_tiles(inkling_swa_head_bias * heads, uint8_t * visible,
                                                 inkling_relative_bias bias, int nq, int nk) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && threadIdx.x < 24) {
        heads[threadIdx.x] = {{bias, int(threadIdx.x)}, visible, int(gridDim.x)};
    }
    bool any = false;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nk) {
        const int q0 = blockIdx.y * inkling_tile_cols;
        for (int q = q0 + threadIdx.y; q < min(nq, q0 + inkling_tile_cols); q += blockDim.y) {
            const int64_t idx = int64_t(q) * nk + k;
            const float m = bias.base_mask ? __half2float(bias.base_mask[idx]) : bias.base_mask_f32[idx];
            if (m != -INFINITY) {
                any = true;
                break;
            }
        }
    }
    const int block_visible = __syncthreads_or(any);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        visible[blockIdx.y * gridDim.x + blockIdx.x] = block_visible != 0;
    }
}

template<typename Head>
static void inkling_launch_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst, Head * heads) {
    ggml_tensor mask_view = *dst->src[3];
    mask_view.data = heads;
    mask_view.nb[2] = sizeof(Head);
    ggml_tensor output = *dst;
    output.src[3] = &mask_view;
    launch_mma_f16_case<128, 128, inkling_tile_cols, 1, Head>(ctx, &output);
}

static __global__ void inkling_set_head_bias(inkling_head_bias * heads, inkling_relative_bias bias) {
    heads[threadIdx.x] = {bias, int(threadIdx.x)};
}

void ggml_hip_inkling_relative_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_tensor * table, const ggml_tensor * ids, const ggml_tensor * mask) {
    const inkling_relative_bias bias = {
        (const float *) table->data, (const int32_t *) ids->data,
        mask->type == GGML_TYPE_F16 ? (const half *) mask->data : nullptr,
        mask->type == GGML_TYPE_F32 ? (const float *) mask->data : nullptr,
        int(table->nb[2]/4), int(table->nb[1]/4), int(ids->nb[1]/4),
    };
    static const bool wmma = [] {
        const char * setting = std::getenv("INKLING_ATTN_WMMA");
        return setting && strcmp(setting, "1") == 0;
    }();
    if (wmma) {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        // Backend copies may add a device prefix and copy index to the graph input name.
        if (GGML_CUDA_CC_IS_RDNA3_0(cc) && strstr(ids->name, "attn_inp_rel_pos_swa") != nullptr) {
            const int tile_keys = ggml_cuda_fattn_mma_get_nbatch_fa(128, 128, inkling_tile_cols, cc);
            const int nk = mask->ne[0], nq = mask->ne[1];
            const dim3 grid((nk + tile_keys - 1) / tile_keys, (nq + inkling_tile_cols - 1) / inkling_tile_cols);
            ggml_cuda_pool_alloc<uint8_t> visible(ctx.pool(), size_t(grid.x) * grid.y);
            ggml_cuda_pool_alloc<inkling_swa_head_bias> heads(ctx.pool(), 24);
            inkling_prepare_swa_tiles<<<grid, dim3(tile_keys, 4), 0, ctx.stream()>>>(heads.get(), visible.get(), bias, nq, nk);
            inkling_launch_wmma(ctx, dst, heads.get());
        } else if (GGML_CUDA_CC_IS_RDNA3_0(cc) && strstr(ids->name, "attn_inp_rel_pos") != nullptr) {
            const int width = ggml_cuda_fattn_mma_get_nbatch_fa(128, 128, inkling_tile_cols, cc);
            const int nk = mask->ne[0], nq = mask->ne[1];
            const int key_tiles = (nk + width - 1) / width;
            ggml_cuda_pool_alloc<uint8_t> far(ctx.pool(), size_t(key_tiles) * nq);
            ggml_cuda_pool_alloc<inkling_global_head_bias> heads(ctx.pool(), 24);
            inkling_prepare_far_tiles<<<dim3(key_tiles, (nq + 7) / 8), dim3(32, 8), 0, ctx.stream()>>>(
                heads.get(), far.get(), bias, nk, nq, width, table->ne[1] - 1);
            inkling_launch_wmma(ctx, dst, heads.get());
        } else {
            ggml_cuda_pool_alloc<inkling_head_bias> heads(ctx.pool(), 24);
            inkling_set_head_bias<<<1, 24, 0, ctx.stream()>>>(heads.get(), bias);
            inkling_launch_wmma(ctx, dst, heads.get());
        }
        static const bool announced = [] { GGML_LOG_INFO("INKLING_WMMA_ACTIVE\n"); return true; }();
        GGML_UNUSED(announced);
        return;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int nwarps = ggml_cuda_fattn_tile_get_nthreads(128, 128, 64, cc) / 32;
    const int batch = ggml_cuda_fattn_tile_get_nbatch_fa(128, 128, 64, cc);
    const fattn_kernel_t dense = flash_attn_tile<128, 128, 64, 1, false>;
    launch_fattn<128, 64, 1>(ctx, dst, flash_attn_tile<128, 128, 64, 1, false, inkling_relative_bias>,
                             nwarps, 0, batch, true, true, false, false, 32, false, dense, bias);
}
#endif

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default:
            GGML_ABORT("Unsupported head size");
    }
}
