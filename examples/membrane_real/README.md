# Membrane Sensor

## Additional Dependencies
1. Deepxde's geometry offers various 2D/3D geometries of closed domains.
```
    git clone git@github.com:yswi/deepxde.git
    cd deepxde
    pip install -e .
    cd ..
```
2. pytorch-volumetric
```
    git clone https://github.com/UM-ARM-Lab/pytorch_volumetric.git
    cd pytorch_volumetric
    pip install -e .
    cd ..
```

## Dataset
Download dataset [here](https://www.dropbox.com/scl/fo/68fmp2rbz2fa8q4m7yls9/AOFd-ABY2rZ9359qCNl9qVk?rlkey=7mss68pwvo93pgcsdme20xrbc&st=0xb1rmd0&dl=0) and update your train config (e.g., configs/train_cfg.py) `config.data_dir` to be \< download path/nisp_membrane_data\>.

## Training
Update the config `config.data_idx=<npy data index>`, `config.noise_lev=<depth noise level>`, and `config.gpu_id=<gpu_id>`.
```
python3 main.py ./configs/train_cfg.py --mode=train
```

## Evaluation
Once trained, to obtain the final predicted error and visualizations, run jupyter notebook examples 'visualization.ipynb' and 'stat.ipynb'.

