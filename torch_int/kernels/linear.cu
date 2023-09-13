#include "include/linear.h"
#include "include/common.h"
#include "include/A8W4_gemm.h"
#include "torch_int/kernels/include/cutlass_extensions/include/cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale.h"
#include "torch_int/kernels/include/cutlass_extensions/include/cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"
#include "torch_int/kernels/include/cutlass_extensions/include/cutlass_extensions/epilogue_helpers.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>


#define THREAD_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

  return optimal_block_num;
}



// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bbf16_obf16(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // BF16
                                       float alpha   //
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = cutlass::bfloat16_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput* const> (out.data_ptr()), LayoutOutput::packed(output_size));
      
  ElementComputeEpilogue alpha_c = ElementComputeEpilogue(alpha);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha_c}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}


// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp16_ofp16(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // BF16
                                       float alpha   //
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = cutlass::half_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput* const> (out.data_ptr()), LayoutOutput::packed(output_size));

  ElementComputeEpilogue alpha_c = ElementComputeEpilogue(alpha); 

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha_c}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}


// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // BF16
                                       float alpha   // 
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = float;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,ElementComputeEpilogue,cutlass::epilogue::thread::ScaleType::NoBetaScaling>, //cutlass::epilogue::thread::ScaleType::NoBetaScaling
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput* const> (out.data_ptr()), LayoutOutput::packed(output_size));

  ElementComputeEpilogue alpha_c = ElementComputeEpilogue(alpha);
  
  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha_c}, 1}; //reinterpret_cast<ElementComputeEpilogue* const> (alpha.data_ptr())
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}
