# A Fair Comparison in Deep Learning-Based QSM Reconstruction
This repository contains the source code for training/inference of QSMnet used in the paper [**"A Fair Comparison in Deep Learning-Based QSM Reconstruction"**](https://doi.org/10.1109/ACCESS.2025.3620427).

## Conda environment
Create the conda environment using the provided YAML file:
```bash
conda env create -n qsmnet2 -f environment.yml
```

## QSMnet training 
### 1. Generate patches for deep learning
```bash
python patch.py
```
### 2. Training the network 
To modify hyperparameters, update the default values in the argument parser within "train.py".
```bash
python train.py -g 0 -s ../save/qsmnet111
```
### 3. Inference the network
To modify hyperparameters, update the default values in the argument parser within "test.py".
```bash
python test.py -g 0 -s ../save/qsmnet111/result -e 25
```
### 4. Calculate evaluation metrics (NRMSE, SSIM, PSNR, HFEN)
```bash
python testmetric.py -p ../network/qsmnet111/result
```
