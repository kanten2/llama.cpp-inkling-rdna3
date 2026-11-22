#include "common.cuh"
#include "fattn-tile.cuh"

#ifdef GGML_USE_HIP
#include "fattn-mma-f16.cuh"

struct inkling_head_bias {
    inkling_relative_bias bias;
    int head;

    __device__ half read(int q, int k, int stride) const {
        return fattn_tile_bias(nullptr, stride, q, k, head, bias);
    }
};

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
        ggml_cuda_pool_alloc<inkling_head_bias> heads(ctx.pool(), 24);
        inkling_set_head_bias<<<1, 24, 0, ctx.stream()>>>(heads.get(), bias);
        ggml_tensor mask_view = *dst->src[3];
        mask_view.data = heads.get();
        mask_view.nb[2] = sizeof(inkling_head_bias);
        ggml_tensor output = *dst;
        output.src[3] = &mask_view;
        launch_mma_f16_case<128, 128, 64, 1, inkling_head_bias>(ctx, &output);
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
