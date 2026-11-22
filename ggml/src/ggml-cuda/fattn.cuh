#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
void ggml_hip_inkling_relative_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_tensor * table, const ggml_tensor * ids, const ggml_tensor * mask);
#endif

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
