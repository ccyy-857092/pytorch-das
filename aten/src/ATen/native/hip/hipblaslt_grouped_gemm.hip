// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#undef __HIP_NO_HALF_CONVERSIONS__
#include <ATen/native/hip/hipblaslt_grouped_gemm.h>
#include <ATen/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/hip/HIPStream.h>

#include <limits>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define TORCH_HIPBLASLT_CHECK(EXPR)               \
  do {                                            \
    hipblasStatus_t __err = EXPR;                 \
    TORCH_CHECK(__err == HIPBLAS_STATUS_SUCCESS,  \
                "hipblaslt error: ",              \
                hipblasStatusToString(__err),     \
                " when calling `" #EXPR "`");     \
  } while (0)

#define TORCH_HIP_CHECK(EXPR)                     \
  do {                                            \
    hipError_t __err = EXPR;                      \
    TORCH_CHECK(__err == hipSuccess,              \
                "hip error: ",                    \
                hipGetErrorString(__err));        \
  } while (0)

namespace at {
namespace hip {
namespace detail {

namespace {

// RAII wrapper for hipblaslt handle cleanup
struct HipBlasLtHandleDeleter {
  void operator()(hipblasLtHandle_t handle) const {
    if (handle != nullptr) {
      hipblasLtDestroy(handle);
    }
  }
};

using HipBlasLtHandlePtr = std::unique_ptr<std::remove_pointer_t<hipblasLtHandle_t>, HipBlasLtHandleDeleter>;

hipblasLtHandle_t get_hipblaslt_handle(int device, hipStream_t /*stream*/) {
  static thread_local auto handles =
      std::map<int, HipBlasLtHandlePtr>();

  auto it = handles.find(device);
  if (it != handles.end()) {
    return it->second.get();
  }
  hipblasLtHandle_t handle = nullptr;
  TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle));
  handles.emplace(device, HipBlasLtHandlePtr(handle));
  return handle;
}

hipDataType hip_type_from_scalar_type(c10::ScalarType type) {
  switch (type) {
    case c10::ScalarType::Half:
      return HIP_R_16F;
    case c10::ScalarType::BFloat16:
      return HIP_R_16BF;
    case c10::ScalarType::Float:
      return HIP_R_32F;
    default:
      TORCH_CHECK(false, "unsupported grouped GEMM dtype: ", type);
  }
}

enum class BlasMatrixLayout {
  RowMajor,
  ColMajor,
  Unsupported,
};

BlasMatrixLayout classify_blas_matrix_layout(
    int64_t rows,
    int64_t cols,
    int64_t row_stride,
    int64_t col_stride) {
  if (rows <= 0 || cols <= 0) {
    return BlasMatrixLayout::Unsupported;
  }
  if (col_stride == 1 && row_stride >= cols) {
    return BlasMatrixLayout::RowMajor;
  }
  if (row_stride == 1 && col_stride >= rows) {
    return BlasMatrixLayout::ColMajor;
  }
  return BlasMatrixLayout::Unsupported;
}

int64_t blas_leading_dim(
    BlasMatrixLayout layout,
    int64_t row_stride,
    int64_t col_stride) {
  return layout == BlasMatrixLayout::RowMajor ? row_stride : col_stride;
}

#if (defined(USE_ROCM) && ROCM_VERSION >= 70000)
#define AT_HIPBLASLT_EXT_HAS_ACCESSORS 1
#else
#define AT_HIPBLASLT_EXT_HAS_ACCESSORS 0
#endif

void set_epilogue_bias(
    hipblaslt_ext::GemmEpilogue& epilogue,
    hipDataType bias_type) {
#if AT_HIPBLASLT_EXT_HAS_ACCESSORS
  epilogue.setMode(HIPBLASLT_EPILOGUE_BIAS);
  epilogue.setBiasDataType(bias_type);
#else
  epilogue.mode = HIPBLASLT_EPILOGUE_BIAS;
  epilogue.bias_data_type = bias_type;
#endif
}

void set_gemm_inputs(
    hipblaslt_ext::GemmInputs& inputs,
    const void* a,
    const void* b,
    const void* c,
    const void* d,
    const void* alpha,
    const void* beta,
    const void* bias) {
#if AT_HIPBLASLT_EXT_HAS_ACCESSORS
  inputs.setA(a);
  inputs.setB(b);
  inputs.setC(c);
  inputs.setD(d);
  inputs.setAlpha(alpha);
  inputs.setBeta(beta);
  inputs.setBias(bias);
#else
  inputs.a = a;
  inputs.b = b;
  inputs.c = c;
  inputs.d = d;
  inputs.alpha = alpha;
  inputs.beta = beta;
  inputs.bias = bias;
#endif
}

hipblasOperation_t op_for_layout(BlasMatrixLayout layout) {
  return layout == BlasMatrixLayout::RowMajor ? HIPBLAS_OP_N : HIPBLAS_OP_T;
}

// Which logical dimension varies from group to group. This is all that is left
// of the user-facing 2D/3D distinction once it reaches the hipblaslt layer.
enum class LenAxis : uint8_t { M, N, K, None };

// Per-group pointer motion, in elements:
//   offset = stride_id * group_id + stride_start * group_start
// A 3D operand steps once per group (stride_id); a 2D operand slides along the
// grouped axis by the cumulative offset (stride_start). Every layout is one of
// those two, so no operand needs a special case.
struct PerGroupOffset {
  int64_t stride_id;
  int64_t stride_start;
};

constexpr int64_t kEmptyGroupScratchSlot = 16;
constexpr int64_t kEmptyGroupScratchStride = 4 * kEmptyGroupScratchSlot;

// The unified description handed to the hipblaslt layer. The four user-level
// dim combinations collapse into: which of M/N/K is per-group, and how each of
// the three operands is offset for a group. hipblaslt takes A = mat_b and
// B = mat_a, so its m is our N and its n is our M.
struct GemmDesc {
  char* mat_a_base;
  char* mat_b_base;
  char* out_base;
  int64_t mat_a_elem;
  int64_t mat_b_elem;
  int64_t out_elem;
  PerGroupOffset mat_a_off;
  PerGroupOffset mat_b_off;
  PerGroupOffset out_off;
  int64_t M;  // fixed dims; the one named by len_axis is per-group instead
  int64_t N;
  int64_t K;
  int64_t total_group;  // total extent along len_axis
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int groups;
  LenAxis len_axis;
  hipblasOperation_t transa;
  hipblasOperation_t transb;
  hipDataType a_type;  // hipblaslt A = mat_b
  hipDataType b_type;  // hipblaslt B = mat_a
  hipDataType d_type;
  int device;
};

// Fills in everything that does not depend on the layout case.
GemmDesc make_desc(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& out) {
  GemmDesc d{};
  d.mat_a_base = static_cast<char*>(mat_a.data_ptr());
  d.mat_b_base = static_cast<char*>(mat_b.data_ptr());
  d.out_base = static_cast<char*>(out.data_ptr());
  d.mat_a_elem = mat_a.element_size();
  d.mat_b_elem = mat_b.element_size();
  d.out_elem = out.element_size();
  d.a_type = hip_type_from_scalar_type(mat_b.scalar_type());
  d.b_type = hip_type_from_scalar_type(mat_a.scalar_type());
  d.d_type = hip_type_from_scalar_type(out.scalar_type());
  d.device = mat_a.get_device();
  return d;
}

// Builds UserArguments entirely on device: no host template, no HTOD. The
// kernel writes every field the picked hipblaslt algo reads (dims, pointers,
// all 8 strides, alpha). Any field it does not touch is left at zero because
// the whole UserArguments slot is zero-initialised at the top of the kernel
// - so beta stays 0, bias/scale/activation flags stay 0/null. This matches the
// pattern used by the reference BlasltGroupedGemmImpl in /groupgg.hip.
template <typename OffsetT>
__global__ void BuildDeviceUserArguments(
    uint32_t count,
    hipblaslt_ext::UserArguments* args,
    const OffsetT* offs,
    GemmDesc d,
    void* bias,
    int bias_type,
    void* empty_scratch) {
  const uint32_t id = blockIdx.x;
  if (id >= count || threadIdx.x != 0) {
    return;
  }

  static_assert(
      sizeof(hipblaslt_ext::UserArguments) % sizeof(uint32_t) == 0,
      "UserArguments must be 32-bit copyable");
  uint32_t* dst = reinterpret_cast<uint32_t*>(&args[id]);
  constexpr uint32_t word_count =
      sizeof(hipblaslt_ext::UserArguments) / sizeof(uint32_t);
  for (uint32_t w = 0; w < word_count; ++w) {
    dst[w] = 0;
  }

  int64_t start = 0;
  int64_t len = 0;
  if (d.len_axis != LenAxis::None) {
    start = id == 0 ? 0 : static_cast<int64_t>(offs[id - 1]);
    len = static_cast<int64_t>(offs[id]) - start;
  }

  int64_t m_run = d.M;
  int64_t n_run = d.N;
  int64_t k_run = d.K;
  switch (d.len_axis) {
    case LenAxis::M:
      m_run = len;
      break;
    case LenAxis::N:
      n_run = len;
      break;
    case LenAxis::K:
      k_run = len;
      break;
    case LenAxis::None:
      break;
  }

  const int64_t gid = static_cast<int64_t>(id);
  char* hb_a = d.mat_b_base +
      (d.mat_b_off.stride_id * gid + d.mat_b_off.stride_start * start) *
          d.mat_b_elem;
  char* hb_b = d.mat_a_base +
      (d.mat_a_off.stride_id * gid + d.mat_a_off.stride_start * start) *
          d.mat_a_elem;
  char* hb_c = d.out_base +
      (d.out_off.stride_id * gid + d.out_off.stride_start * start) * d.out_elem;

  // alpha bytes for 1.0f (compute type is fp32). beta stays at zero from the
  // zero-init above, so D = alpha * A @ B (no accumulation into C).
  constexpr uint32_t kOneF32Bits = 0x3f800000u;
  for (uint32_t byte = 0; byte < sizeof(float); ++byte) {
    args[id].alpha[byte] =
        static_cast<int8_t>((kOneF32Bits >> (byte * 8)) & 0xffu);
  }

  if (m_run <= 0 || n_run <= 0 || k_run <= 0) {
    char* scratch =
        static_cast<char*>(empty_scratch) + gid * kEmptyGroupScratchStride;
    for (int64_t byte = 0; byte < kEmptyGroupScratchStride; ++byte) {
      scratch[byte] = 0;
    }
    args[id].m = 1;
    args[id].n = 1;
    args[id].batch = 1;
    args[id].k = 1;
    args[id].a = scratch;
    args[id].b = scratch + kEmptyGroupScratchSlot;
    args[id].c = scratch + 2 * kEmptyGroupScratchSlot;
    args[id].d = scratch + 3 * kEmptyGroupScratchSlot;
    args[id].strideA1 = 1;
    args[id].strideA2 = 1;
    args[id].strideB1 = 1;
    args[id].strideB2 = 1;
    args[id].strideC1 = 1;
    args[id].strideC2 = 1;
    args[id].strideD1 = 1;
    args[id].strideD2 = 1;
    return;
  }

  // hipblaslt swaps A/B (A = mat_b, B = mat_a), so hipblaslt m = our N,
  // hipblaslt n = our M. Fill every field the algo can read.
  args[id].m = static_cast<uint32_t>(n_run);
  args[id].n = static_cast<uint32_t>(m_run);
  args[id].batch = 1;
  args[id].k = static_cast<uint32_t>(k_run);
  args[id].a = hb_a;
  args[id].b = hb_b;
  args[id].c = hb_c;
  args[id].d = hb_c;
  args[id].strideA1 = static_cast<uint32_t>(d.lda);
  args[id].strideA2 = static_cast<uint32_t>(n_run * k_run);
  args[id].strideB1 = static_cast<uint32_t>(d.ldb);
  args[id].strideB2 = static_cast<uint32_t>(k_run * m_run);
  args[id].strideC1 = static_cast<uint32_t>(d.ldc);
  args[id].strideC2 = static_cast<uint32_t>(n_run * m_run);
  args[id].strideD1 = static_cast<uint32_t>(d.ldc);
  args[id].strideD2 = static_cast<uint32_t>(n_run * m_run);

  if (bias != nullptr) {
    args[id].bias = bias;
    args[id].biasType = bias_type;
  }
}

void run_gemm(
    const GemmDesc& d,
    const at::Tensor* offs,
    const void* bias,
    c10::ScalarType bias_scalar_type,
    hipStream_t stream) {
  TORCH_CHECK(d.groups > 0, "grouped GEMM requires at least one group");
  TORCH_CHECK(
      d.M <= std::numeric_limits<uint32_t>::max() &&
          d.N <= std::numeric_limits<uint32_t>::max() &&
          d.K <= std::numeric_limits<uint32_t>::max() &&
          d.total_group <= std::numeric_limits<uint32_t>::max(),
      "grouped GEMM dims exceed UserArguments range");

  hipblaslt_ext::GemmEpilogue epilogue;
  int bias_type = 0;
  if (bias != nullptr) {
    const hipDataType type = hip_type_from_scalar_type(bias_scalar_type);
    TORCH_CHECK(
        type == HIP_R_32F || type == HIP_R_16BF,
        "grouped GEMM bias must be float or bfloat16");
    set_epilogue_bias(epilogue, type);
    bias_type = static_cast<int>(type);
  }

  // Placeholder problem sizes: the heuristic only needs a representative shape,
  // the kernel rewrites each group's real m/n/k at run time. Put the whole
  // grouped extent on group 0 and 1 on the rest -- a uniform split makes
  // hipblaslt pick a fixed-tile batched algo that then breaks on uneven groups.
  const int64_t safe_m = d.M > 0 ? d.M : 1;
  const int64_t safe_n = d.N > 0 ? d.N : 1;
  const int64_t safe_k = d.K > 0 ? d.K : 1;
  const int64_t safe_tg = d.total_group > 0 ? d.total_group : 1;

  // hipblaslt m corresponds to our N and its n to our M, so the grouped axis
  // lands in the swapped vector.
  std::vector<int64_t> m(d.groups, safe_n);
  std::vector<int64_t> n(d.groups, safe_m);
  std::vector<int64_t> k(d.groups, safe_k);
  std::vector<int64_t>* grouped = nullptr;
  switch (d.len_axis) {
    case LenAxis::M:
      grouped = &n;
      break;
    case LenAxis::N:
      grouped = &m;
      break;
    case LenAxis::K:
      grouped = &k;
      break;
    case LenAxis::None:
      break;
  }
  if (grouped != nullptr) {
    std::fill(grouped->begin(), grouped->end(), 1);
    (*grouped)[0] = safe_tg;
  }

  // Batch strides follow the same substitution: the grouped dim is the total.
  const int64_t dim_m = d.len_axis == LenAxis::M ? safe_tg : safe_m;
  const int64_t dim_n = d.len_axis == LenAxis::N ? safe_tg : safe_n;
  const int64_t dim_k = d.len_axis == LenAxis::K ? safe_tg : safe_k;

  std::vector<int64_t> batch(d.groups, 1);
  std::vector<int64_t> lda_v(d.groups, d.lda);
  std::vector<int64_t> ldb_v(d.groups, d.ldb);
  std::vector<int64_t> ldc_v(d.groups, d.ldc);
  std::vector<int64_t> ldd_v(d.groups, d.ldc);
  std::vector<int64_t> stride_a(d.groups, dim_k * dim_n);
  std::vector<int64_t> stride_b(d.groups, dim_k * dim_m);
  std::vector<int64_t> stride_c(d.groups, dim_n * dim_m);
  std::vector<int64_t> stride_d(d.groups, dim_n * dim_m);

  std::vector<hipblaslt_ext::GemmInputs> inputs(d.groups);
  float one = 1.0f;
  float zero = 0.0f;
  for (auto& input : inputs) {
    set_gemm_inputs(
        input,
        d.mat_b_base,
        d.mat_a_base,
        d.out_base,
        d.out_base,
        &one,
        &zero,
        bias);
  }

  hipblaslt_ext::GroupedGemm gemm(
      get_hipblaslt_handle(d.device, stream),
      d.transa,
      d.transb,
      d.a_type,
      d.b_type,
      d.d_type,
      d.d_type,
      HIPBLAS_COMPUTE_32F);
  hipblaslt_ext::GemmProblemType problem_type{
      d.transa, d.transb, d.a_type, d.b_type, d.d_type, d.d_type,
      HIPBLAS_COMPUTE_32F};
  std::vector<hipblaslt_ext::GemmEpilogue> epilogues{epilogue};
  TORCH_HIPBLASLT_CHECK(gemm.setProblem(
      m,
      n,
      k,
      batch,
      lda_v,
      ldb_v,
      ldc_v,
      ldd_v,
      stride_a,
      stride_b,
      stride_c,
      stride_d,
      epilogues,
      inputs,
      problem_type));

  hipblaslt_ext::GemmPreference preference;
  preference.setMaxWorkspaceBytes(0);
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  TORCH_HIPBLASLT_CHECK(gemm.algoGetHeuristic(1, preference, results));
  TORCH_CHECK(!results.empty(), "no valid grouped GEMM algorithm found");
  TORCH_HIPBLASLT_CHECK(
      gemm.initialize(results[0].algo, nullptr, true, stream));

  // Per-call device buffers. at::empty uses torch's caching allocator so hipMalloc
  // is amortised - we do not maintain a separate cache. The tensors are stream-tagged
  // on release so the allocator will not hand them back to us until the async work
  // on `stream` has drained.
  auto opts =
      at::TensorOptions().dtype(at::kByte).device(at::kCUDA, d.device);
  at::Tensor args_buf = at::empty(
      {static_cast<int64_t>(d.groups) *
       static_cast<int64_t>(sizeof(hipblaslt_ext::UserArguments))},
      opts);
  at::Tensor scratch_buf =
      at::empty({d.groups * kEmptyGroupScratchStride}, opts);
  auto* args =
      reinterpret_cast<hipblaslt_ext::UserArguments*>(args_buf.data_ptr());

  // No host template, no H2D memcpy: the kernel constructs UserArguments from
  // scratch on device. Any UserArguments field the kernel does not touch stays
  // zero because the kernel zero-inits the whole slot.
  auto launch = [&](auto* offsets) {
    using OffsetT = std::remove_pointer_t<decltype(offsets)>;
    hipLaunchKernelGGL(
        (BuildDeviceUserArguments<OffsetT>),
        dim3(d.groups),
        dim3(1),
        0,
        stream,
        static_cast<uint32_t>(d.groups),
        args,
        offsets,
        d,
        const_cast<void*>(bias),
        bias_type,
        scratch_buf.data_ptr());
    TORCH_HIP_CHECK(hipGetLastError());
  };

  if (d.len_axis == LenAxis::None) {
    launch(static_cast<const int*>(nullptr));
  } else {
    TORCH_CHECK(
        offs != nullptr && offs->defined() && offs->dim() == 1,
        "grouped GEMM requires 1D offsets");
    TORCH_CHECK(offs->is_cuda(), "offsets must be CUDA tensor");
    // The kernel indexes offs as a raw pointer.
    TORCH_CHECK(offs->is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(
        offs->size(0) == d.groups,
        "grouped GEMM: offs length (",
        offs->size(0),
        ") must equal group count (",
        d.groups,
        ")");
    if (offs->scalar_type() == c10::kInt) {
      launch(offs->data_ptr<int>());
    } else {
      TORCH_CHECK(
          offs->scalar_type() == c10::kLong,
          "offsets must be int32 or int64");
      launch(offs->data_ptr<int64_t>());
    }
  }

  TORCH_HIPBLASLT_CHECK(gemm.run(args, stream));
}

}  // anonymous namespace

void hipblaslt_grouped_gemm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias_opt,
    at::Tensor& out) {
  TORCH_CHECK(
      mat_a.is_cuda() && mat_b.is_cuda() && out.is_cuda(),
      "grouped GEMM requires CUDA tensors");
  TORCH_CHECK(
      mat_a.get_device() == mat_b.get_device() &&
          mat_a.get_device() == out.get_device(),
      "grouped GEMM inputs and output must be on the same device");
  TORCH_CHECK(
      mat_a.scalar_type() == mat_b.scalar_type() &&
          mat_a.scalar_type() == out.scalar_type(),
      "grouped GEMM requires matching dtypes");

  const c10::Device device = mat_a.device();
  const c10::DeviceGuard device_guard(device);

  const bool use_bias = bias_opt.has_value() && bias_opt->defined();
  const void* bias = use_bias ? bias_opt->data_ptr() : nullptr;
  if (use_bias) {
    TORCH_CHECK(bias_opt->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(
        bias_opt->get_device() == mat_a.get_device(),
        "bias must be on the same device as inputs");
  }
  const auto bias_scalar_type =
      use_bias ? bias_opt->scalar_type() : c10::kFloat;

  hipStream_t stream = c10::hip::getCurrentHIPStream(device.index());

  at::Tensor out_contig = out.contiguous();
  const bool need_out_copy = (out_contig.data_ptr() != out.data_ptr());

  const int a_dim = mat_a.dim();
  const int b_dim = mat_b.dim();

  const at::Tensor* offs_ptr = (offs.has_value() && offs->defined())
      ? &offs.value()
      : nullptr;
  if (offs_ptr != nullptr) {
    TORCH_CHECK(
        offs_ptr->get_device() == mat_a.get_device(),
        "offsets must be on the same device as inputs");
  }

  GemmDesc d = make_desc(mat_a, mat_b, out_contig);

  if (a_dim == 2 && b_dim == 3) {
    // MoE forward: A[Mtot, K] @ B[G, K, N] -> out[Mtot, N], grouped on M.
    TORCH_CHECK(offs_ptr != nullptr, "2d/3d grouped GEMM requires offsets");
    const BlasMatrixLayout a_layout = classify_blas_matrix_layout(
        mat_a.size(0), mat_a.size(1), mat_a.stride(0), mat_a.stride(1));
    const BlasMatrixLayout b_layout = classify_blas_matrix_layout(
        mat_b.size(1), mat_b.size(2), mat_b.stride(1), mat_b.stride(2));
    TORCH_CHECK(
        a_layout != BlasMatrixLayout::Unsupported &&
            b_layout != BlasMatrixLayout::Unsupported,
        "unsupported layout in 2d/3d");

    d.len_axis = LenAxis::M;
    d.groups = static_cast<int>(mat_b.size(0));
    d.N = mat_b.size(2);
    d.K = mat_a.size(1);
    d.total_group = mat_a.size(0);
    d.mat_a_off = {0, mat_a.stride(0)};
    d.mat_b_off = {mat_b.stride(0), 0};
    d.out_off = {0, out_contig.stride(0)};
    d.lda = blas_leading_dim(b_layout, mat_b.stride(1), mat_b.stride(2));
    d.ldb = blas_leading_dim(a_layout, mat_a.stride(0), mat_a.stride(1));
    d.ldc = out_contig.stride(0);
    d.transa = op_for_layout(b_layout);
    d.transb = op_for_layout(a_layout);
  } else if (a_dim == 2 && b_dim == 2) {
    // Grouped along K: A[M, Ktot] @ B[Ktot, N] -> out[G, M, N]. Empty groups own
    // dedicated MxN output slots that the kernel routes to scratch, so zero the
    // output up front to make those slots defined.
    TORCH_CHECK(offs_ptr != nullptr, "2d/2d grouped GEMM requires offsets");
    TORCH_CHECK(
        out_contig.dim() == 3,
        "2d/2d grouped GEMM requires 3D out [G, M, N]");
    out_contig.zero_();
    const BlasMatrixLayout a_layout = classify_blas_matrix_layout(
        mat_a.size(0), mat_a.size(1), mat_a.stride(0), mat_a.stride(1));
    const BlasMatrixLayout b_layout = classify_blas_matrix_layout(
        mat_b.size(0), mat_b.size(1), mat_b.stride(0), mat_b.stride(1));
    TORCH_CHECK(
        a_layout != BlasMatrixLayout::Unsupported &&
            b_layout != BlasMatrixLayout::Unsupported,
        "unsupported layout in 2d/2d");

    d.len_axis = LenAxis::K;
    d.groups = static_cast<int>(out_contig.size(0));
    d.M = mat_a.size(0);
    d.N = mat_b.size(1);
    d.total_group = mat_a.size(1);
    d.mat_a_off = {0, mat_a.stride(1)};
    d.mat_b_off = {0, mat_b.stride(0)};
    d.out_off = {out_contig.stride(0), 0};
    d.lda = blas_leading_dim(b_layout, mat_b.stride(0), mat_b.stride(1));
    d.ldb = blas_leading_dim(a_layout, mat_a.stride(0), mat_a.stride(1));
    d.ldc = d.N;
    d.transa = op_for_layout(b_layout);
    d.transb = op_for_layout(a_layout);
  } else if (a_dim == 3 && b_dim == 2) {
    // Grouped along N: A[G, M, K] @ B[K, Ntot] -> out[M, Ntot].
    TORCH_CHECK(offs_ptr != nullptr, "3d/2d grouped GEMM requires offsets");
    const BlasMatrixLayout a_layout = classify_blas_matrix_layout(
        mat_a.size(1), mat_a.size(2), mat_a.stride(1), mat_a.stride(2));
    const BlasMatrixLayout b_layout = classify_blas_matrix_layout(
        mat_b.size(0), mat_b.size(1), mat_b.stride(0), mat_b.stride(1));
    TORCH_CHECK(
        a_layout != BlasMatrixLayout::Unsupported &&
            b_layout != BlasMatrixLayout::Unsupported,
        "unsupported layout in 3d/2d");

    d.len_axis = LenAxis::N;
    d.groups = static_cast<int>(mat_a.size(0));
    d.M = mat_a.size(1);
    d.K = mat_a.size(2);
    d.total_group = mat_b.size(1);
    d.mat_a_off = {mat_a.stride(0), 0};
    d.mat_b_off = {0, mat_b.stride(1)};
    d.out_off = {0, out_contig.stride(1)};
    d.lda = blas_leading_dim(b_layout, mat_b.stride(0), mat_b.stride(1));
    d.ldb = blas_leading_dim(a_layout, mat_a.stride(1), mat_a.stride(2));
    // Groups write disjoint N slices of one [M, Ntot] output, so the row pitch is
    // the full Ntot, not the slice width.
    d.ldc = d.total_group;
    d.transa = op_for_layout(b_layout);
    d.transb = op_for_layout(a_layout);
  } else if (a_dim == 3 && b_dim == 3) {
    // Batched: A[G, M, K] @ B[G, K, N] -> out[G, M, N], no grouped axis.
    const BlasMatrixLayout a_layout = classify_blas_matrix_layout(
        mat_a.size(1), mat_a.size(2), mat_a.stride(1), mat_a.stride(2));
    const BlasMatrixLayout b_layout = classify_blas_matrix_layout(
        mat_b.size(1), mat_b.size(2), mat_b.stride(1), mat_b.stride(2));
    TORCH_CHECK(
        a_layout != BlasMatrixLayout::Unsupported &&
            b_layout != BlasMatrixLayout::Unsupported,
        "unsupported layout in 3d/3d");
    TORCH_CHECK(
        mat_a.size(0) == mat_b.size(0),
        "3d/3d grouped GEMM requires matching batch dim");

    d.len_axis = LenAxis::None;
    d.groups = static_cast<int>(mat_a.size(0));
    d.M = mat_a.size(1);
    d.K = mat_a.size(2);
    d.N = mat_b.size(2);
    d.total_group = 1;
    d.mat_a_off = {mat_a.stride(0), 0};
    d.mat_b_off = {mat_b.stride(0), 0};
    d.out_off = {out_contig.stride(0), 0};
    d.lda = blas_leading_dim(b_layout, mat_b.stride(1), mat_b.stride(2));
    d.ldb = blas_leading_dim(a_layout, mat_a.stride(1), mat_a.stride(2));
    d.ldc = out_contig.stride(1);
    d.transa = op_for_layout(b_layout);
    d.transb = op_for_layout(a_layout);
  } else {
    TORCH_CHECK(
        false,
        "grouped GEMM: unsupported dim combination a=",
        a_dim,
        " b=",
        b_dim);
  }

  run_gemm(d, offs_ptr, bias, bias_scalar_type, stream);

  if (need_out_copy) {
    out.copy_(out_contig);
  }
}

}
}
}

#undef TORCH_HIPBLASLT_CHECK
#undef TORCH_HIP_CHECK
