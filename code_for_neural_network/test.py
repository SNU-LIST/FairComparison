## 1. parse input
## 2. initiate
## 3. train with 중간그림들 저장
## 4. test
## 5. testmeric
## 6. retrain
import argparse, os, json
from cmath import inf
import numpy as np
import pandas as pd
import torch, random
from util import *
from network import *
from importdata import *
from scipy.ndimage import rotate
from datetime import datetime

## 1. parse input
if 1:
    parser = argparse.ArgumentParser(description='compareqsmnet arguments')
    parser.add_argument('-g', '--GPU',      type=int,    required=True,  help='GPU device to use (0, 1, ...)')
    parser.add_argument('-s', '--save_dir', type=str,    required=True,  help='directory where inferenced maps are stored')
    parser.add_argument('-e', '--net_epoch',type=int,    required=True,  help='the training epoch of the network')    

    parser.add_argument('--TRAIN_DATA',    type=str,    default='../data/D111',   help='path for training data')
    parser.add_argument('--TEST_DATA',    type=str,    default='../data/D111/result',   help='path for validation data')

    # hyperparameters
    parser.add_argument('--NET_CHA',       type=int,    default=32)
    parser.add_argument('--NET_KER',       type=int,    default=5)
    parser.add_argument('--NET_ACT',       type=str,    default='leaky_relu')
    parser.add_argument('--NET_SLP',       type=float,  default=0.1)
    parser.add_argument('--NET_POOL',      type=str,    default='max')
    parser.add_argument('--NET_LAY',       type=int,    default=4)
    parser.add_argument('--TRAIN_EPOCH',   type=int,    default=25)
    parser.add_argument('--MAX_STEP',      type=int,    default=inf)
    parser.add_argument('--TRAIN_BATCH',   type=int,    default=12)
    parser.add_argument('--TRAIN_LR',      type=float,  default=0.001)
    parser.add_argument('--TRAIN_W1',      type=float,  default=0.5)
    parser.add_argument('--TRAIN_W2',      type=float,  default=0.1)
    parser.add_argument('--TRAIN_OPT',     type=str,    default='RMSProp')

    # settings for reproducibility
    parser.add_argument('--BATCH_ORDER_SEED',   type=int,    default=0)
    parser.add_argument('--WEIGHT_INIT_SEED',   type=int,    default=0)
    parser.add_argument('--CUDA_deterministic', type=str,    default='True')
    parser.add_argument('--CUDA_benchmark',     type=str,    default='False')

    args = parser.parse_args()
    if os.path.exists(f'{args.save_dir}/../CONFIG.txt'):
        with open(f'{args.save_dir}/../CONFIG.txt','r') as f:
            CONFIG=json.loads(f.read())
        args.NET_CHA = CONFIG['NET_CHA']
        args.NET_KER = CONFIG['NET_KER']
        args.NET_ACT = CONFIG['NET_ACT']
        args.NET_SLP = CONFIG['NET_SLP']
        args.NET_POOL = CONFIG['NET_POOL']
        args.NET_LAY = CONFIG['NET_LAY']
        args.TRAIN_BATCH = CONFIG['TRAIN_BATCH']
        args.TRAIN_LR = CONFIG['TRAIN_LR']
        args.TRAIN_W1 = CONFIG['TRAIN_W1']
        args.TRAIN_W2 = CONFIG['TRAIN_W2']
        args.TRAIN_OPT = CONFIG['TRAIN_OPT']
    print_options(parser,args)
    args = vars(args)

## 2. initiate
if 2:
    os.makedirs(args["save_dir"], exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['GPU'])
    device=torch.device("cuda")

    os.environ['PYTHONHASHSEED'] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if args['CUDA_deterministic'] == 'True':
        torch.backends.cudnn.deterministic=True
    elif args['CUDA_deterministic'] == 'False':
        torch.backends.cudnn.deterministic=False
    else:
        print('wrong args[CUDA_deterministic]')
        raise ValueError
    if args['CUDA_benchmark'] == 'True':
        torch.backends.cudnn.benchmark=True
    elif args['CUDA_benchmark'] == 'False':
        torch.backends.cudnn.benchmark=False
    else:
        print('wrong args[CUDA_benchmark]')
        raise ValueError


# 3. load data
dt = load_traindata(args['TRAIN_DATA'],load_scale=True)
input_mean = dt['input_mean']; input_std = dt['input_std']; 
label_mean = dt['label_mean']; label_std = dt['label_std'];

model = load_network(args).eval().to(device)
load_weights(model,f'{args["save_dir"]}/../model/ep{args["net_epoch"]:03d}.pth')

dt = load_testdata(args["TEST_DATA"])
for t in range(dt['num']):
    tfield = dt['tfield'][t]
    tlabel = dt['tlabel'][t]
    tmask  = dt['tmask'][t]
    tpred  = np.empty(dt['matrix_size'])
    with torch.no_grad():
        for ori in range(dt['matrix_size'][-1]):
            x = torch.tensor(tfield[np.newaxis,np.newaxis,...,ori], device=device, dtype=torch.float)
            if args["TEST_DATA"].split('/')[-1]=='D113':
                pad = torch.zeros((x.shape[0],x.shape[1],x.shape[2],x.shape[3],5), device=device, dtype=torch.float)
                x = torch.cat((pad,x,pad), 4)
                
            x = ( x - input_mean ) / input_std
            pred = model(x).squeeze().cpu().numpy()*label_std+label_mean 
            if args["TEST_DATA"].split('/')[-1]=='D113':
                pred = pred[...,5:-5]            
            tpred[...,ori] = pred*tmask[...,ori]

#             dtimg = display_data(); sl=dt['slice'];
#             dtimg.figsize = (12,8)
#             dtimg.data.append(rotate(tfield[...,sl,ori], -90)); dtimg.v.append((-0.05,0.05)); dtimg.label.append('field')
#             dtimg.data.append(rotate(tpred[ ...,sl,ori], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('predict')
#             dtimg.data.append(rotate(tlabel[...,sl,ori], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('cosmos')
#             fig=display_images(1, 3, dtimg, p=False)
#             fig.savefig(f'{TEST_DATA_FOLDER}/sub{t+1}_ori{ori}.png')                  
        scipy.io.savemat(f'{args["save_dir"]}/sub{t+1}.mat', mdict={'pred':tpred})
    
