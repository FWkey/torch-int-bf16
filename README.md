# torch-int
This repository contains integer operators on GPUs for PyTorch. Define linear_a8_w8_bXX_oXX with XX selected from 'FP32' 'BF16' 'FP16'. 'FP16' is not recommended because of possible overflow.

## Dependencies
- CUTLASS
- PyTorch with CUDA 11.3
- NVIDIA-Toolkit 11.3
- CUDA Driver 11.3
- gcc g++ 9.4.0
- cmake >= 3.12

## Installation
```bash
git clone --recurse-submodules https://github.com/FWkey/torch-int-bf16.git
conda create -n int python=3.8
conda activate int
conda install -c anaconda gxx_linux-64=9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
source environment.sh
bash build_cutlass.sh
python setup.py install
```

## Test
```bash
python tests/test_linear_modules.py
```
