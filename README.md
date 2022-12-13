# compareQSMnet
This is a (not completed) source code utilized for experiments of "A fair comparison in deep learning-based MRI reconstruction"

## Requirements 
* Conda (https://docs.conda.io/en/latest/)
* MATLAB 2016b (https://www.mathworks.com/products/matlab.html)
* NVIDIA TITAN Xp GPU

## Usage
### Step 0: First time only
1. Create conda environment
```bash
cd code
conda create --name qsmnet -c conda-forge -c anaconda --file requirements.txt 
```

### Step 1: dataset generation (MATLAB & Python)
1. dataset generation & augmentation (matlab)  
```bash
cd code/dataset_generation
matlab
```  
 For generation D111,
```bash
cd ../../test1
healthy_run_D111
run2_D111
run3_D111
```
For generation D113, 
```bash
cd ../../test1
healthy_run_D113
run2_D113
run3_D113
```
For generation DVSHARP, DPDF, and DLBV,
```bash
cd ../../test1
healthy_run_DVSHARP
run2_Dbkg
run2_DVSHARP
run3_DVSHARP
```
3. generation patches for deep learning (python)  
```bash
conda activate qsmnet
python patch.py
```

### Step 2: QSMnet training & inference (Python)
1. repository & activate conda environment
```bash
cd code
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
