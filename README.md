# compareQSMnet
This is a (not completed) source code utilized for experiments of "A fair comparison in deep learning-based MRI reconstruction"

## Requirements 
* Conda (https://docs.conda.io/en/latest/)
* MATLAB 2016b (https://www.mathworks.com/products/matlab.html)
* NVIDIA TITAN Xp GPU
* CUDA 10.0

## Usage
### Step 0: First time only
1. Create conda environment
```bash
conda create --name qsmnet -c conda-forge -c anaconda -c pytorch --file requirements.txt 
```
2. data structure: please construct data with below structures
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
2. generation patches for deep learning (python)  
```bash
cd code_for_dataset_generation
conda activate qsmnet
python patch.py
```

### Step 2: QSMnet training & inference (Python)
1. repository & activate conda environment
```bash
cd code_for_neural_network
conda activate qsmnet
```
2. QSMnet training  
If you want to change hyperparameters, please replace the default value in parser arguments of "train.py"
```bash
python train.py -g 0 -s ../network/qsmnet111
```
3. QSMnet inference  
If you want to change hyperparameters, please replace the default value in parser arguments of "test.py"
```bash
python test.py -g 0 -s ../network/qsmnet111/test -e 25
```
4. calculate metrics (NRMSE, SSIM, PSNR, HFEN) of inferenced maps
```bash
python testmetric.py -p ../network/qsmnet111/test??
```
