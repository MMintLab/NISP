# NISP: Neural Inverse Source Problems

This is an official git repository of [Neural Inverse Source Problems](https://arxiv.org/abs/2411.01665) presented in CoRL 2024. We appreciate [Jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi) for offering a great codebase.


## Installation

Ensure that you have Python 3.8 or later installed on your system.
Our code is GPU-only.
We highly recommend using the most recent versions of JAX and JAX-lib, along with compatible CUDA and cuDNN versions.
The code has been tested and confirmed to work with the following versions:

- JAX 0.4.16
- CUDA 12.1
- cuDNN 8.9


``` 
git clone https://github.com/MMintLab/NISP
cd NISP
pip install -e .
```

## Quickstart

Please follow the readme.md in each example folder. For instance,

``` 
cd examples/membrane_real
``` 
To train the model, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/train.py
```



