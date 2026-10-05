#include "fill.cuh"
#include "convert.cuh"

#include <cstdint>

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

// 4 elements per thread through a single aligned vector store (16 bytes for f32,
// 8 bytes for f16); requires dst to be aligned to sizeof(fill_vec4<T>).
template <typename T>
struct alignas(4*sizeof(T)) fill_vec4 {
    T v[4];
};

template <typename T>
static __global__ void fill_kernel_v4(T * dst, const int64_t k, const T value) {
    const int64_t i0 = 4*((int64_t)blockDim.x * blockIdx.x + threadIdx.x);
    if (i0 + 4 <= k) {
        fill_vec4<T> v;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v.v[j] = value;
        }
        *(fill_vec4<T> *)(dst + i0) = v;
    } else {
        // last partial vector: 1-3 scalar elements
        for (int64_t i = i0; i < k; ++i) {
            dst[i] = value;
        }
    }
}

template <typename T>
static void fill_cuda(T * dst, const int64_t k, const T value, cudaStream_t stream) {
    if ((uintptr_t) dst % sizeof(fill_vec4<T>) == 0) {
        const int64_t nvec = (k + 3) / 4;
        const int64_t num_blocks = (nvec + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;
        fill_kernel_v4<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>(dst, k, value);
        return;
    }
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;
    fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>(dst, k, value);
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_cuda((float *)dst_d, k, value, stream);
            break;
        case GGML_TYPE_F16:
            fill_cuda((half *)dst_d, k, ggml_cuda_cast<half>(value), stream);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
