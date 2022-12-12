# compareQSMnet
This is a (not completed) source code utilized for experiments of "A fair comparison in deep learning-based MRI reconstruction"

## Requirements 
* Conda
* NVIDIA TITAN Xp GPU

## Dataset Generation

## Usage: First time only
1. Create conda environment
```bash
cd code
conda create --name qsmnet -c conda-forge -c anaconda --file requirements.txt 
```

## Usage 1: dataset generation (MATLAB & Python)
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

## Usage 2: QSMnet training & inference (Python)
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
