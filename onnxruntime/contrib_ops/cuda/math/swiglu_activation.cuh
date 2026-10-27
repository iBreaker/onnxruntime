// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The clamped SwiGLU of one element, shared by every kernel that fuses it:
//   G = min(gate, limit), L = clamp(up, -limit, limit)         (clamps off when limit <= 0)
//   result = G * sigmoid(alpha * G) * (L + beta)
// Evaluated in float, widened to double for the product when alpha * G is below -87 so that a negative tail is not
// lost to float exp underflow. Keep the arithmetic identical to the SwiGLU operator.
__device__ __forceinline__ float SwiGluActivation(float gate, float up, float limit, float alpha, float beta) {
  float g = gate;
  float l = up;
  if (limit > 0.0f) {
    g = fminf(g, limit);
    l = fminf(fmaxf(l, -limit), limit);
  }
  const float scaled = alpha * g;
  if (scaled < -87.0f) {
    // Float exp enters its subnormal range near -87 and underflows to zero by -104, which would zero a tail that
    // the following multiply by (l + beta) can bring back into range. Keep the whole product wide.
    const double wide_g = static_cast<double>(g);
    const double e = exp(wide_g * static_cast<double>(alpha));
    return static_cast<float>(wide_g * e / (1.0 + e) * (static_cast<double>(l) + static_cast<double>(beta)));
  }
  // Keeps the exponent non-positive without cancelling small negative-input results.
  const float sigmoid = scaled > 0.0f ? 1.0f / (1.0f + expf(-scaled)) : expf(scaled) / (1.0f + expf(scaled));
  return g * sigmoid * (l + beta);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
