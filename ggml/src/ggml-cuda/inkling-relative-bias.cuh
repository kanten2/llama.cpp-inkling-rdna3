#pragma once

struct inkling_relative_bias {
    const float * table;
    const int32_t * rows;
    const half * base_mask;
    const float * base_mask_f32;
    int table_query_stride;
    int table_row_stride;
    int index_stride;
};

static __device__ __forceinline__ half fattn_tile_bias(const half *, int stride, int q, int k, int head,
                                                       inkling_relative_bias bias) {
    const int row = bias.rows[q * bias.index_stride + k];
    const float value = bias.table[q * bias.table_query_stride + row * bias.table_row_stride + head];
    const int index = q * stride + k;
    const float mask = bias.base_mask ? __half2float(bias.base_mask[index]) : bias.base_mask_f32[index];
    return __float2half_rn(value + mask);
}
