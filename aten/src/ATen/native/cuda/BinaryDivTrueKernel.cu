#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/BFloat16.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace at::native {
namespace binary_internal {

constexpr char div_name[] = "div_kernel";

#ifdef USE_ROCM
namespace {

// The generic ROCm path already performs FP32 division for BFloat16.  On
// gfx938, convert two FP32 quotients back to BFloat16 with one packed VALU
// instruction instead of two scalar conversion/rounding sequences.
constexpr int kBf16DivThreads = 512;
constexpr int kBf16DivVectorsPerThread = 4;
constexpr int kBf16DivElementsPerVector = 8;
constexpr int kBf16DivElementsPerBlock =
    kBf16DivThreads * kBf16DivVectorsPerThread * kBf16DivElementsPerVector;
constexpr int64_t kBf16DivMinNumel = 4LL * 1024 * 1024;

// On gfx938 this path is enabled by default.  Setting this environment
// variable to 1 bypasses the runtime architecture check for bring-up and
// A/B tests; non-gfx938 device builds use the scalar conversion fallback.
inline bool bf16_div_pk_force_enabled() {
  if (const char* e = std::getenv("PYTORCH_ROCM_FAST_DIV")) {
    if (e[0] == '0' && e[1] == '\0') return false;   // explicit off
    if (std::atoi(e) != 0) return true;             // explicit on
  }
  c10::DeviceIndex current_device = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&current_device));
  return at::detail::getCUDAHooks().isGPUArch({"gfx938"}, current_device);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } value{static_cast<uint32_t>(bits) << 16};
  return value.f;
}

__device__ __forceinline__ uint32_t cvt_pk_bf16(float lo, float hi) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx938__)
  uint32_t out;
  // DTK exposes no C++ builtin for this gfx938 instruction.
  asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(out) : "v"(lo), "v"(hi));
  return out;
#else
  // Other targets still compile this kernel when building a fat binary.
  // Keep their device pass free of the gfx938-only packed instruction.
  const uint32_t lo_bits = static_cast<uint32_t>(c10::BFloat16(lo).x);
  const uint32_t hi_bits = static_cast<uint32_t>(c10::BFloat16(hi).x);
  return lo_bits | (hi_bits << 16);
#endif
}

__device__ __forceinline__ uint32_t div_bf16_pair(uint32_t av, uint32_t bv) {
  const float q0 = bf16_bits_to_float(static_cast<uint16_t>(av)) /
      bf16_bits_to_float(static_cast<uint16_t>(bv));
  const float q1 = bf16_bits_to_float(static_cast<uint16_t>(av >> 16)) /
      bf16_bits_to_float(static_cast<uint16_t>(bv >> 16));
  return cvt_pk_bf16(q0, q1);
}

__global__ __launch_bounds__(kBf16DivThreads) void div_true_bf16_pk_kernel(
    const int4* __restrict__ a,
    const int4* __restrict__ b,
    int4* __restrict__ out) {
#pragma unroll
  for (int chunk = 0; chunk < kBf16DivVectorsPerThread; ++chunk) {
    const int index =
        (blockIdx.x * kBf16DivVectorsPerThread + chunk) * blockDim.x +
        threadIdx.x;
    const int4 av = __ldg(a + index);
    const int4 bv = __ldg(b + index);
    int4 ov;
    ov.x = static_cast<int>(div_bf16_pair(
        static_cast<uint32_t>(av.x), static_cast<uint32_t>(bv.x)));
    ov.y = static_cast<int>(div_bf16_pair(
        static_cast<uint32_t>(av.y), static_cast<uint32_t>(bv.y)));
    ov.z = static_cast<int>(div_bf16_pair(
        static_cast<uint32_t>(av.z), static_cast<uint32_t>(bv.z)));
    ov.w = static_cast<int>(div_bf16_pair(
        static_cast<uint32_t>(av.w), static_cast<uint32_t>(bv.w)));
    out[index] = ov;
  }
}

inline bool aligned16(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 15U) == 0;
}

inline bool try_div_true_bf16_pk(TensorIteratorBase& iter) {
  const int64_t n = iter.numel();
  const void* out_ptr = iter.data_ptr(0);
  const void* a_ptr = iter.data_ptr(1);
  const void* b_ptr = iter.data_ptr(2);
  // make sure address isn't conflict
  bool addr_conf = out_ptr == a_ptr || out_ptr == b_ptr || a_ptr == b_ptr;
  const bool candidate = iter.common_dtype() == kBFloat16 &&
      iter.is_contiguous() && !iter.is_cpu_scalar(1) &&
      !iter.is_cpu_scalar(2) && n >= kBf16DivMinNumel &&
      !addr_conf &&
      n <= std::numeric_limits<int>::max() &&
      n % kBf16DivElementsPerBlock == 0 && aligned16(out_ptr) &&
      aligned16(a_ptr) && aligned16(b_ptr);
  if (!candidate) {
    return false;
  }

  if (!bf16_div_pk_force_enabled()) {
    return false;
  }

  const dim3 grid(static_cast<uint32_t>(n / kBf16DivElementsPerBlock));
  div_true_bf16_pk_kernel<<<
      grid,
      kBf16DivThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const int4*>(iter.data_ptr(1)),
      static_cast<const int4*>(iter.data_ptr(2)),
      static_cast<int4*>(iter.data_ptr(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

} // namespace
#endif

void div_true_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto div_string = jiterator_stringify(
        template <typename T> T div_kernel(T a, T b) { return a / b; });
    opmath_jitted_gpu_kernel_with_scalars<div_name, scalar_t, scalar_t>(
        iter, div_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
#endif
    return;
  }
#ifdef USE_ROCM
  if (try_div_true_bf16_pk(iter)) {
    return;
  }
#endif
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          using high_prec_t = std::conditional_t<
              c10::is_complex<scalar_t>::value,
              c10::complex<double>,
              double>;
          auto inv_b = static_cast<opmath_t>(high_prec_t(1.0) / iter.scalar_value<high_prec_t>(2));
          iter.remove_operand(2);
          gpu_kernel(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          DivFunctor<scalar_t> f;
          gpu_kernel_with_scalars(iter, f);
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_cuda)

} // namespace at::native
