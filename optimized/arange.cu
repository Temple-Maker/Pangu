#include "arange.cuh"

#include <cstdint>

static __global__ void arange_f32(float * dst, const int ne0, const float start, const float step) {
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    dst[nidx] = start + step * nidx;
}

// 4 elements per thread via a single 16-byte store; requires dst to be 16-byte aligned.
static __global__ void arange_f32_v4(float * dst, const int ne0, const float start, const float step) {
    const int i0 = 4*(threadIdx.x + blockIdx.x * blockDim.x);
    if (i0 + 4 <= ne0) {
        const float4 v = make_float4(start + step * (i0 + 0),
                                     start + step * (i0 + 1),
                                     start + step * (i0 + 2),
                                     start + step * (i0 + 3));
        *(float4 *)(dst + i0) = v;
    } else {
        // last partial vector: 1-3 scalar elements
        for (int i = i0; i < ne0; ++i) {
            dst[i] = start + step * i;
        }
    }
}

static void arange_f32_cuda(float * dst, const int ne0, const float start, const float step, cudaStream_t stream) {
    if ((uintptr_t) dst % 16 == 0) {
        const int nvec = (ne0 + 3) / 4;
        const int num_blocks = (nvec + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
        arange_f32_v4<<<num_blocks, CUDA_ARANGE_BLOCK_SIZE, 0, stream>>>(dst, ne0, start, step);
        return;
    }
    int num_blocks = (ne0 + CUDA_ARANGE_BLOCK_SIZE - 1) / CUDA_ARANGE_BLOCK_SIZE;
    arange_f32<<<num_blocks, CUDA_ARANGE_BLOCK_SIZE, 0, stream>>>(dst, ne0, start,  step);
}

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start;
    float stop;
    float step;
    memcpy(&start, (float *)dst->op_params + 0, sizeof(float));
    memcpy(&stop,  (float *)dst->op_params + 1, sizeof(float));
    memcpy(&step,  (float *)dst->op_params + 2, sizeof(float));

    int64_t steps = (int64_t)ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);

    arange_f32_cuda(dst_d, dst->ne[0], start, step, stream);
}
