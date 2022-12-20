# A fair comparison in deep learning MRI reconstruction
This is a source code utilized for experiments of "A fair comparison in deep learning MRI reconstruction"

## Requirements 
* Conda (https://docs.conda.io/en/latest/)
* MATLAB 2016b (https://www.mathworks.com/products/matlab.html)
* FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
* ANTs (https://github.com/ANTsX/ANTs)
* CUDA 10.0
* NVIDIA TITAN Xp GPU

## Usage
### Step 0: First time only
1. Create conda environment
```bash
conda env create -n qsmnet -f requirements.yaml
```
2. Data: please organize the qsmnet data according to the format below
```
data  
  └ D111
      └ train2  
          └ IMG.mat  
      └ train3   
          └ IMG.mat    
      └ ...
  └ D113
  └ ...
```                

### Step 1: dataset generation (MATLAB & Python)
1. dataset generation & augmentation (matlab)  
```bash
cd code_for_dataset_generation
matlab
final_run
```  
2. generate patches for deep learning (python)  
```bash
cd code_for_dataset_generation
conda activate qsmnet
python patch.py
```

### Step 2: QSMnet training & inference (Python)
1. activate conda environment
```bash
cd code_for_neural_network
conda activate qsmnet
```
2. QSMnet training  
If you want to change hyperparameters, please replace the default value in parser arguments of "train.py"
```bash
python train.py -g 0 -s ../save/qsmnet111
```
3. QSMnet inference  
If you want to change hyperparameters, please replace the default value in parser arguments of "test.py"
```bash
python test.py -g 0 -s ../save/qsmnet111/result -e 25
```
4. calculate evaluation metrics (NRMSE, SSIM, PSNR, HFEN) of inferenced maps
```bash
python testmetric.py -p ../network/qsmnet111/result
```
