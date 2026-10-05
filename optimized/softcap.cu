#include "softcap.cuh"

#include <cstdint>

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

// 4 elements per thread via 16-byte accesses; requires x and dst to be 16-byte aligned.
static __global__ void softcap_f32_v4(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i0 = 4*(blockDim.x*blockIdx.x + threadIdx.x);

    ggml_cuda_pdl_sync();
    if (i0 + 4 <= k) {
        float4 v = *(const float4 *)(x + i0);
        v.x = tanhf(scale * v.x) * softcap;
        v.y = tanhf(scale * v.y) * softcap;
        v.z = tanhf(scale * v.z) * softcap;
        v.w = tanhf(scale * v.w) * softcap;
        *(float4 *)(dst + i0) = v;
    } else {
        // last partial vector: 1-3 scalar elements
        for (int i = i0; i < k; ++i) {
            dst[i] = tanhf(scale * x[i]) * softcap;
        }
    }
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    if ((uintptr_t) x % 16 == 0 && (uintptr_t) dst % 16 == 0) {
        const int nvec = (k + 3) / 4;
        const int num_blocks = (nvec + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(softcap_f32_v4, launch_params, x, dst, scale, softcap, k);
        return;
    }
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}
