#include "clamp.cuh"

#include <cstdint>

static __device__ __forceinline__ float op_clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

template <class T>
static __global__ void op_clamp_kernel(const T * x, T * dst, const T min, const T max, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_clamp((float)x[i], (float)min, (float)max);
}

// 4 elements per thread through a single aligned vector access: 16-byte loads/stores for
// float, 8-byte for half. Requires x and dst to be aligned to sizeof(clamp_vec4<T>).
// No __restrict__: clamp may run in place with dst == x.
template <class T>
struct alignas(4*sizeof(T)) clamp_vec4 {
    T v[4];
};

template <class T>
static __global__ void op_clamp_kernel_v4(const T * x, T * dst, const T min, const T max, const int k) {
    const int i0 = 4*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i0 + 4 <= k) {
        clamp_vec4<T> vv = *(const clamp_vec4<T> *)(x + i0);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            vv.v[j] = (T)op_clamp((float)vv.v[j], (float)min, (float)max);
        }
        *(clamp_vec4<T> *)(dst + i0) = vv;
    } else {
        // last partial vector: 1-3 scalar elements
        for (int i = i0; i < k; i++) {
            dst[i] = (T)op_clamp((float)x[i], (float)min, (float)max);
        }
    }
}

template <class T>
static void clamp_cuda(const T * x, T * dst, const T min, const T max, const int k, cudaStream_t stream) {
    if ((uintptr_t) x % sizeof(clamp_vec4<T>) == 0 && (uintptr_t) dst % sizeof(clamp_vec4<T>) == 0) {
        const int nvec = (k + 3) / 4;
        const int num_blocks = (nvec + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
        op_clamp_kernel_v4<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
        return;
    }
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
}


void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        clamp_cuda((const half *)src0_d, (half *)dst_d, (half)min, (half)max, ggml_nelements(src0), stream);
    } else {
        clamp_cuda((const float *)src0_d, (float *)dst_d, (float)min, (float)max, ggml_nelements(src0), stream);
    }
}
