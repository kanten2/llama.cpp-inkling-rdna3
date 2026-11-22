#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

#ifdef GGML_USE_HIP
void ggml_cuda_op_row_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * dst, float threshold);
void ggml_cuda_op_scale_add(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, ggml_tensor * add);
#endif

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
