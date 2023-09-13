#include "include/bmm.h"
#include "include/fused.h"
#include "include/linear.h"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_a8_w8_bfp16_ofp16", &linear_a8_w8_bfp16_ofp16, "Linear (I8-OFP16)");
  m.def("linear_a8_w8_bfp32_ofp32", &linear_a8_w8_bfp32_ofp32, "Linear (I8-OFP32)");
  m.def("linear_a8_w8_bbf16_obf16", &linear_a8_w8_bbf16_obf16, "Linear (I8-OBF16)");
  m.def("dq_add_layernorm_q", &dq_add_layernorm_q,
        "DQ + Add + LayerNorm (INT8)");
  m.def("bmm_s8t_s8n_s8t", &bmm_s8t_s8n_s8t, "BMM (INT8 IO) A x B.T");
  m.def("bmm_s8t_s8n_f32t", &bmm_s8t_s8n_f32t, "BMM (INT8 I FP32 O) A x B.T");
  m.def("bmm_s8t_s8n_s32t", &bmm_s8t_s8n_s32t,
        "BMM (INT8 In Int32 Out) A x B.T");
}
