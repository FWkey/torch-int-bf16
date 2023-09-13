#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>

torch::Tensor linear_a8_w8_bbf16_obf16(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // BF16
                                      float alpha  // 
);

torch::Tensor linear_a8_w8_bfp16_ofp16(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP16
                                      float alpha  // 
);

torch::Tensor linear_a8_w8_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                      float alpha  // 
);

#endif // LINEAR_HS
