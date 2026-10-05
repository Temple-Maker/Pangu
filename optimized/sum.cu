#include "sum.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

#include <cstdint>

#ifndef GGML_CUDA_USE_CUB
#define CUDA_SUM_BLOCK_SIZE 256
#define CUDA_SUM_MAX_BLOCKS 1024

// Grid-stride partial sums with a block-level tree reduction; one output per block.
// Launched a second time over the per-block partials to produce the final scalar.
// The grid geometry is fixed for a given ne, which keeps the reduction order (and
// therefore the result) deterministic run to run, unlike an atomicAdd single pass.
static __global__ void sum_f32_partial(const float * __restrict__ x, float * __restrict__ partial, const int64_t ne) {
    __shared__ float shared_vals[WARP_SIZE];

    float sum = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) gridDim.x*blockDim.x) {
        sum += x[i];
    }

    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}
#endif // !GGML_CUDA_USE_CUB

void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream) {
#ifdef GGML_CUDA_USE_CUB
    size_t tmp_size = 0;
    DeviceReduce::Sum(nullptr,       tmp_size, x, dst, ne, stream);
    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
#else
    // Two-pass tree reduction. The previous fallback summed the whole tensor with a
    // single thread block via sum_rows; this spreads pass 1 over up to
    // CUDA_SUM_MAX_BLOCKS blocks and reduces their partials in a second launch.
    const int64_t nblocks_needed = (ne + CUDA_SUM_BLOCK_SIZE - 1) / CUDA_SUM_BLOCK_SIZE;

    if (nblocks_needed <= 1) { // also covers ne == 0: an empty grid-stride loop writes 0.0f
        sum_f32_partial<<<1, CUDA_SUM_BLOCK_SIZE, 0, stream>>>(x, dst, ne);
        return;
    }

    const int nblocks = (int) MIN(nblocks_needed, (int64_t) CUDA_SUM_MAX_BLOCKS);

    ggml_cuda_pool_alloc<float> partial_alloc(pool, nblocks);
    sum_f32_partial<<<nblocks, CUDA_SUM_BLOCK_SIZE, 0, stream>>>(x, partial_alloc.ptr, ne);
    sum_f32_partial<<<1,       CUDA_SUM_BLOCK_SIZE, 0, stream>>>(partial_alloc.ptr, dst, nblocks);
#endif // GGML_CUDA_USE_CUB
}

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguously_allocated(src0));

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne = ggml_nelements(src0);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
}
