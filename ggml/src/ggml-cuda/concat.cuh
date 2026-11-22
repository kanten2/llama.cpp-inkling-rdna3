#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
bool ggml_cuda_try_concat_history(ggml_backend_cuda_context & ctx, ggml_tensor * window, ggml_tensor * copy);
#endif
