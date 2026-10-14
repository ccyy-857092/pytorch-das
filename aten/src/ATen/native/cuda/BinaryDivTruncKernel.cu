#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <cstdint>
#include <type_traits>

#ifdef USE_ROCM
#include <cstdlib>        // std::getenv, std::atoi
#endif

namespace at::native {
namespace binary_internal {

// gfx938 fast path for contiguous int8/uint8 div(..., rounding_mode="trunc").
// A reciprocal plus a signed 2^-10 bias is exact over the complete 8-bit
// domain after truncation, including the GPU's integer divide-by-zero result.
constexpr int kDivTruncThreads = 64;
constexpr int kDivTruncBytesPerThread = 64;
constexpr int64_t kDivTruncElementsPerBlock =
    kDivTruncThreads * kDivTruncBytesPerThread;
constexpr int64_t kDivFastPathThreshold = 2048 * 1024; // 限定2M

#ifdef USE_ROCM
// Env-var name for forcing the int8 fast path on/off for other arch.
//   PYTORCH_ROCM_FAST_DIV=0  -> force-disable
//   PYTORCH_ROCM_FAST_DIV=1  -> force-enable
//   (unset)          -> enable only when running on gfx938
inline bool div_trunc_int8_fast_arch_enabled() {
  if (const char* e = std::getenv("PYTORCH_ROCM_FAST_DIV")) {
    if (e[0] == '0' && e[1] == '\0') return false;   // explicit off
    if (std::atoi(e) != 0) return true;             // explicit on
  }

  // Runtime gate: the AMD reciprocal and floating-point builtins compile on
  // all ROCm targets, but this fast path is enabled by default only on gfx938,
  // where its correctness and performance have been validated.
  c10::DeviceIndex current_device = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&current_device));
  return at::detail::getCUDAHooks().isGPUArch({"gfx938"}, current_device);
}
#endif

template <typename T>
__device__ __forceinline__ int32_t unpack_div_trunc_byte(uint32_t value);

template <>
__device__ __forceinline__ int32_t
unpack_div_trunc_byte<int8_t>(uint32_t value) {
  return static_cast<int32_t>(static_cast<int8_t>(value));
}

template <>
__device__ __forceinline__ int32_t
unpack_div_trunc_byte<uint8_t>(uint32_t value) {
  return static_cast<int32_t>(static_cast<uint8_t>(value));
}

__device__ __forceinline__ int32_t div_trunc_byte(int32_t a, int32_t b) {
#if defined(__HIP_PLATFORM_AMD__)
  float quotient =
      static_cast<float>(a) * __builtin_amdgcn_rcpf(static_cast<float>(b));
#else
  float quotient = static_cast<float>(a) * __frcp_rn(static_cast<float>(b));
#endif
  quotient += __builtin_copysignf(1.0f / 1024.0f, quotient);
  // Convert through int32_t before extracting the low byte. Direct
  // float-to-int8/uint8 conversion is not reliable on all HIP compilers.
  return static_cast<int32_t>(__builtin_truncf(quotient));
}

template <typename T>
__device__ __forceinline__ uint32_t div_trunc_pack4(uint32_t a, uint32_t b) {
  const int32_t q0 =
      div_trunc_byte(unpack_div_trunc_byte<T>(a), unpack_div_trunc_byte<T>(b));
  const int32_t q1 = div_trunc_byte(
      unpack_div_trunc_byte<T>(a >> 8), unpack_div_trunc_byte<T>(b >> 8));
  const int32_t q2 = div_trunc_byte(
      unpack_div_trunc_byte<T>(a >> 16), unpack_div_trunc_byte<T>(b >> 16));
  const int32_t q3 = div_trunc_byte(
      unpack_div_trunc_byte<T>(a >> 24), unpack_div_trunc_byte<T>(b >> 24));
  return static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
}

template <typename T>
__device__ __forceinline__ int4 div_trunc_pack16(int4 a, int4 b) {
  int4 out;
  out.x = static_cast<int>(div_trunc_pack4<T>(
      static_cast<uint32_t>(a.x), static_cast<uint32_t>(b.x)));
  out.y = static_cast<int>(div_trunc_pack4<T>(
      static_cast<uint32_t>(a.y), static_cast<uint32_t>(b.y)));
  out.z = static_cast<int>(div_trunc_pack4<T>(
      static_cast<uint32_t>(a.z), static_cast<uint32_t>(b.z)));
  out.w = static_cast<int>(div_trunc_pack4<T>(
      static_cast<uint32_t>(a.w), static_cast<uint32_t>(b.w)));
  return out;
}

template <typename T>
__global__ __launch_bounds__(kDivTruncThreads) void div_trunc_int8_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out) {
  const int64_t block_base =
      static_cast<int64_t>(blockIdx.x) * kDivTruncElementsPerBlock;

#pragma unroll
  for (int pack = 0; pack < kDivTruncBytesPerThread / 16; ++pack) {
    const int64_t offset = block_base +
        (static_cast<int64_t>(pack) * kDivTruncThreads + threadIdx.x) * 16;
    const int4 av = __ldg(reinterpret_cast<const int4*>(a + offset));
    const int4 bv = __ldg(reinterpret_cast<const int4*>(b + offset));
    *reinterpret_cast<int4*>(out + offset) = div_trunc_pack16<T>(av, bv);
  }
}

inline bool div_trunc_aligned_16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15U) == 0;
}

template <typename T>
inline bool try_div_trunc_int8_fast(TensorIteratorBase& iter) {
  const int64_t numel = iter.numel();
  const void* out_ptr = iter.data_ptr(0);
  const void* a_ptr = iter.data_ptr(1);
  const void* b_ptr = iter.data_ptr(2);
  // make sure address isn't conflict
  bool addr_conf = out_ptr == a_ptr || out_ptr == b_ptr || a_ptr == b_ptr;
  if (!iter.is_contiguous() || iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2) ||
      numel <= 0 || numel % kDivTruncElementsPerBlock != 0 ||
      numel < kDivFastPathThreshold ||
      addr_conf ||
      !div_trunc_aligned_16(out_ptr) ||
      !div_trunc_aligned_16(a_ptr) ||
      !div_trunc_aligned_16(b_ptr)) {
    return false;
  }

#ifdef USE_ROCM
  // Arch gate: default to gfx938 only; override via PYTORCH_ROCM_FAST_DIV.
  if (!div_trunc_int8_fast_arch_enabled()) {
    return false;
  }
#endif

  const dim3 grid(static_cast<uint32_t>(numel / kDivTruncElementsPerBlock));
  div_trunc_int8_kernel<T>
      <<<grid, dim3(kDivTruncThreads), 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const T*>(iter.data_ptr(1)),
          static_cast<const T*>(iter.data_ptr(2)),
          static_cast<T *>(iter.data_ptr(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

void div_trunc_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
#ifdef USE_ROCM
  if (dtype == kChar && try_div_trunc_int8_fast<int8_t>(iter)) {
    return;
  }
  if (dtype == kByte && try_div_trunc_int8_fast<uint8_t>(iter)) {
    return;
  }
#endif
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
          iter.remove_operand(2);
          gpu_kernel(iter, [inv_b] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return std::trunc(a * inv_b);
          });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          gpu_kernel_with_scalars(
              iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return std::trunc(a / b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_trunc_stub, &binary_internal::div_trunc_kernel_cuda)

} // namespace at::native
