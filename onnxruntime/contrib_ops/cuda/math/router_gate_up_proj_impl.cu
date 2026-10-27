// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/router_gate_up_proj_impl.h"

#include <limits>

#include "contrib_ops/cuda/math/router_gate_up_proj_kernel.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kSplitThreads = 256;

__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ void FromFloat(float v, __nv_bfloat16& out) { out = __float2bfloat16_rn(v); }
__device__ __forceinline__ void FromFloat(float v, __half& out) { out = __float2half_rn(v); }

// Generic path epilogue: copies the router columns and applies the activation to the gate/up columns of each row.
template <typename V>
__global__ void SplitActivationKernel(int64_t rows, int n, int router, int inter, float limit, float alpha, float beta,
                                      const V* __restrict__ y, V* __restrict__ router_out, V* __restrict__ act_out) {
  const int64_t total = rows * (router + inter);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int64_t row = i / (router + inter);
    const int col = static_cast<int>(i - row * (router + inter));
    const V* yr = y + row * n;
    if (col < router) {
      router_out[row * router + col] = yr[col];
    } else {
      const int j = col - router;
      const float gate = ToFloat(yr[router + j]);
      const float up = ToFloat(yr[router + inter + j]);
      FromFloat(SwiGluActivation(gate, up, limit, alpha, beta), act_out[row * inter + j]);
    }
  }
}

template <typename V>
void LaunchSplitActivation(cudaStream_t stream, int64_t rows, int64_t n, int64_t router, int64_t inter, float limit,
                           float alpha, float beta, const void* y, void* router_out, void* act_out) {
  const int64_t total = rows * (router + inter);
  const int64_t want = (total + kSplitThreads - 1) / kSplitThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  SplitActivationKernel<V><<<blocks, kSplitThreads, 0, stream>>>(
      rows, static_cast<int>(n), static_cast<int>(router), static_cast<int>(inter), limit, alpha, beta,
      static_cast<const V*>(y), static_cast<V*>(router_out), static_cast<V*>(act_out));
}

}  // namespace

bool RouterGateUpProjUsesFusedKernel(int sm_major, int64_t rows, int64_t k, int64_t router, int64_t inter) {
  constexpr int64_t kMaxDim = std::numeric_limits<int>::max() / 2;
  return sm_major >= 9 && rows >= 1 && rows <= router_gate_up::kMaxRows && k <= kMaxDim && router <= kMaxDim &&
         inter <= kMaxDim &&
         router_gate_up::Supports(static_cast<int>(rows), static_cast<int>(k), static_cast<int>(router),
                                  static_cast<int>(inter));
}

Status LaunchRouterGateUpProjFused(cudaStream_t stream, bool use_pdl, bool is_bf16, const void* x, const void* w,
                                   void* router_out, void* act_out, int64_t rows, int64_t k, int64_t n,
                                   int64_t router, int64_t inter, float limit, float alpha, float beta) {
  router_gate_up::Params params{};
  params.x = static_cast<const uint16_t*>(x);
  params.w = static_cast<const uint16_t*>(w);
  params.router_out = static_cast<uint16_t*>(router_out);
  params.act_out = static_cast<uint16_t*>(act_out);
  params.rows = static_cast<int>(rows);
  params.k = static_cast<int>(k);
  params.n = static_cast<int>(n);
  params.router = static_cast<int>(router);
  params.inter = static_cast<int>(inter);
  params.limit = limit;
  params.alpha = alpha;
  params.beta = beta;
  CUDA_RETURN_IF_ERROR(router_gate_up::Launch(params, stream, use_pdl, is_bf16));
  return Status::OK();
}

Status LaunchRouterGateUpProjGeneric(cublasHandle_t cublas, cudaStream_t stream, bool is_bf16, const void* x,
                                     const void* w, void* scratch, void* router_out, void* act_out, int64_t rows,
                                     int64_t k, int64_t n, int64_t router, int64_t inter, float limit, float alpha,
                                     float beta) {
  ORT_RETURN_IF_NOT(rows <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
                        k <= std::numeric_limits<int>::max(),
                    "RouterGateUpProj dimension is too large");
  const float one = 1.0f;
  const float zero = 0.0f;
  const cudaDataType_t type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;
  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(rows),
                                      static_cast<int>(k), &one, w, type, static_cast<int>(n), x, type,
                                      static_cast<int>(k), &zero, scratch, type, static_cast<int>(n),
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  if (is_bf16) {
    LaunchSplitActivation<__nv_bfloat16>(stream, rows, n, router, inter, limit, alpha, beta, scratch, router_out,
                                         act_out);
  } else {
    LaunchSplitActivation<__half>(stream, rows, n, router, inter, limit, alpha, beta, scratch, router_out, act_out);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
