import torch
from torch_int._CUDA import linear_a8_w8_bfp32_ofp32,linear_a8_w8_bbf16_obf16
from icecream import ic



@torch.no_grad()
def test_quant_linear_a8_w8_bbf16_obf16():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    qscales = torch.randn(N, dtype=torch.float32)
    clampv = 10.
    bias = torch.randn(N, dtype=torch.float32)*0.01
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * qscales.view(-1,1)
    linear.bias.data = bias.float() * 0.01
    y_gt = linear(x.float()/127.*clampv)
    y = linear_a8_w8_bbf16_obf16( input8.to(torch.int8), weight_d, bias/qscales.view(-1), clampv/127.)*qscales.view(1,-1)
    print(y.shape)
    ic(torch.allclose(y_gt, y.cpu(), atol=0.5))


if __name__ == '__main__':
    print('test_quant_linear_a8_w8_bbf16_obf16')
    test_quant_linear_a8_w8_bbf16_obf16()

