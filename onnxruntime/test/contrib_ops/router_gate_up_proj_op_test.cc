// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

struct RouterGateUpCase {
  int64_t rows;
  int64_t k;
  int64_t router;
  int64_t inter;
  float limit = 0.0f;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool weight_is_initializer = true;
};

template <typename T>
float RoundToT(float v) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(v).ToFloat();
  } else {
    return MLFloat16(v).ToFloat();
  }
}

template <typename T>
std::vector<T> ToT(const std::vector<float>& v) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(v);
  } else {
    return ToFloat16(v);
  }
}

float SwiGluRef(float g, float l, float limit, float alpha, float beta) {
  if (limit > 0.0f) {
    g = std::min(g, limit);
    l = std::min(std::max(l, -limit), limit);
  }
  return g / (1.0f + std::exp(-alpha * g)) * (l + beta);
}

template <typename T>
void RunRouterGateUpProjTest(const RouterGateUpCase& c, const char* activation = nullptr) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA is not available";
  }

  const int64_t n = c.router + 2 * c.inter;
  std::mt19937 generator(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(c.rows * c.k));
  std::vector<float> w(static_cast<size_t>(c.k * n));
  for (float& v : x) v = RoundToT<T>(dist(generator));
  for (float& v : w) v = RoundToT<T>(dist(generator) * 0.1f);

  std::vector<float> router(static_cast<size_t>(c.rows * c.router));
  std::vector<float> activated(static_cast<size_t>(c.rows * c.inter));
  for (int64_t m = 0; m < c.rows; ++m) {
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int64_t kk = 0; kk < c.k; ++kk) {
      const double xv = x[static_cast<size_t>(m * c.k + kk)];
      for (int64_t j = 0; j < n; ++j) y[static_cast<size_t>(j)] += xv * w[static_cast<size_t>(kk * n + j)];
    }
    for (int64_t j = 0; j < c.router; ++j) {
      router[static_cast<size_t>(m * c.router + j)] = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(j)]));
    }
    for (int64_t j = 0; j < c.inter; ++j) {
      const float gate = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(c.router + j)]));
      const float up = RoundToT<T>(static_cast<float>(y[static_cast<size_t>(c.router + c.inter + j)]));
      activated[static_cast<size_t>(m * c.inter + j)] = RoundToT<T>(SwiGluRef(gate, up, c.limit, c.alpha, c.beta));
    }
  }

  OpTester tester("RouterGateUpProj", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("router_size", c.router);
  if (activation != nullptr) tester.AddAttribute<std::string>("activation", activation);
  tester.AddAttribute<float>("limit", c.limit);
  tester.AddAttribute<float>("activation_alpha", c.alpha);
  tester.AddAttribute<float>("activation_beta", c.beta);
  tester.AddInput<T>("X", {c.rows, c.k}, ToT<T>(x));
  tester.AddInput<T>("W", {c.k, n}, ToT<T>(w), c.weight_is_initializer);
  // 16-bit outputs can differ by one ulp when the fp32 accumulation order differs from the reference.
  tester.AddOutput<T>("router", {c.rows, c.router}, ToT<T>(router), false, 0.02f, 0.02f);
  tester.AddOutput<T>("activated", {c.rows, c.inter}, ToT<T>(activated), false, 0.02f, 0.02f);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

TEST(RouterGateUpProjTest, FusedSingleRow) { RunRouterGateUpProjTest<BFloat16>({1, 512, 128, 64}); }
TEST(RouterGateUpProjTest, FusedOddRows) { RunRouterGateUpProjTest<BFloat16>({3, 1024, 256, 128}); }
TEST(RouterGateUpProjTest, FusedEightRows) { RunRouterGateUpProjTest<BFloat16>({8, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, FusedClampAndBeta) {
  RunRouterGateUpProjTest<BFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f});
}
TEST(RouterGateUpProjTest, FusedDynamicWeight) {
  RunRouterGateUpProjTest<BFloat16>({2, 512, 128, 64, 0.0f, 1.0f, 0.0f, false});
}
TEST(RouterGateUpProjTest, FusedExplicitActivation) { RunRouterGateUpProjTest<BFloat16>({2, 512, 128, 64}, "swiglu"); }
TEST(RouterGateUpProjTest, GenericManyRows) { RunRouterGateUpProjTest<BFloat16>({40, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, GenericUnalignedK) { RunRouterGateUpProjTest<BFloat16>({4, 200, 48, 24}); }

TEST(RouterGateUpProjTest, Float16FusedSingleRow) { RunRouterGateUpProjTest<MLFloat16>({1, 512, 128, 64}); }
TEST(RouterGateUpProjTest, Float16FusedEightRows) { RunRouterGateUpProjTest<MLFloat16>({8, 2560, 512, 640}); }
TEST(RouterGateUpProjTest, Float16FusedClampAndBeta) {
  RunRouterGateUpProjTest<MLFloat16>({5, 1024, 128, 64, 0.5f, 1.702f, 1.0f});
}
TEST(RouterGateUpProjTest, Float16GenericManyRows) { RunRouterGateUpProjTest<MLFloat16>({40, 1024, 128, 64}); }

}  // namespace test
}  // namespace onnxruntime
