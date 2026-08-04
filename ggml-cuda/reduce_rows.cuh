#include "common.cuh"

#include <cstdint>

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm>
static __global__ void reduce_rows_f32(const float * x_ptr, float * dst_ptr, const int ncols) {
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;

    ggml_cuda_pdl_sync();
    if (ncols % 4 == 0 && (uintptr_t) x % 16 == 0) {
        // ncols % 4 == 0 keeps every row base 16-byte aligned, so each thread can fold
        // 4 columns per float4 load; two loads per iteration preserve the 8-deep unroll.
        // Note: lane grouping differs from the scalar path, so the reduction order (and
        // rounding) differs — reductions do not promise an association order.
        const float4 * GGML_CUDA_RESTRICT x4 = (const float4 *) (x + row * ncols);
        const int ncols4 = ncols / 4;

        float4 temp4[num_unroll / 4];
        float4 sum_temp4[num_unroll / 4] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                                             make_float4(0.0f, 0.0f, 0.0f, 0.0f) };
        for (int i = col; i < ncols4;) {
            for (int j = 0; j < num_unroll / 4; ++j) {
                if (i < ncols4) {
                    temp4[j] = x4[i];
                } else {
                    temp4[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                i += blockDim.x;
            }
            for (int j = 0; j < num_unroll / 4; ++j) {
                sum_temp4[j].x += temp4[j].x;
                sum_temp4[j].y += temp4[j].y;
                sum_temp4[j].z += temp4[j].z;
                sum_temp4[j].w += temp4[j].w;
            }
        }
        for (int j = 0; j < num_unroll / 4; ++j) {
            sum += (sum_temp4[j].x + sum_temp4[j].y) + (sum_temp4[j].z + sum_temp4[j].w);
        }
    } else {
        float temp[num_unroll];
        float sum_temp[num_unroll] = { 0.0f };

        for (int i = col; i < ncols;) {
            for (int j = 0; j < num_unroll; ++j) {
                if (i < ncols) {
                    temp[j] = x[row * ncols + i];
                } else {
                    temp[j] = 0;
                }
                i += blockDim.x;
            }
            for (int j = 0; j < num_unroll; ++j) {
                sum_temp[j] += temp[j];
            }
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum += sum_temp[j];
        }
    }

    // sum up partial sums
    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}
