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
    parser.add_argument('-g', '--GPU',     type=int,    required=True)
    parser.add_argument('-s', '--save_dir',type=str,    required=True)

    parser.add_argument('--TRAIN_DATA',    type=str,    default='/fast_storage/chungseok/DVSHARP')
    parser.add_argument('--VAL_DATA',      type=str,    default='/home/chungseok/project/compareqsmnet/data/DVSHARP')

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

    parser.add_argument('--BATCH_ORDER_SEED',   type=int,    default=0)
    parser.add_argument('--WEIGHT_INIT_SEED',   type=int,    default=0)
    parser.add_argument('--CUDA_deterministic', type=str,    default='True')
    parser.add_argument('--CUDA_benchmark',     type=str,    default='False')

    args = parser.parse_args()
    if os.path.exists(f'{args.save_dir}/CONFIG.txt'):
        with open(f'{args.save_dir}/CONFIG.txt','r') as f:
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
    os.makedirs(f'{args["save_dir"]}/model', exist_ok=True)
    os.makedirs(f'{args["save_dir"]}/val', exist_ok=True)

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


dt = load_traindata(args['TRAIN_DATA'])
d = dipole_kernel((64,64,64), dt['voxel_size'], (0,0,1))
model = load_network(args)
if args["TRAIN_OPT"] == "RMSProp":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args["TRAIN_LR"])    
if args["TRAIN_OPT"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args["TRAIN_LR"]) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.95, last_epoch=-1)

logger_train = Logger(f'{args["save_dir"]}/log_train.csv')
logger_val = Logger(f'{args["save_dir"]}/log_val.csv')

data = pd.read_csv(f'../seed_batch/seed{args["BATCH_ORDER_SEED"]}.csv')
if args['TRAIN_DATA'].split('/')[-1]=='D113':
    data = pd.read_csv(f'../seed_batch/seed{args["BATCH_ORDER_SEED"]}_D113.csv')
ind_stack = np.array(data)
num_batch = int(ind_stack.shape[1]/args["TRAIN_BATCH"])

# for H tuning
checkpoint = torch.load(f'../seed_weight/seed{args["WEIGHT_INIT_SEED"]}.pth')
model.load_state_dict(checkpoint['model'])
del(checkpoint); 
torch.cuda.empty_cache();

ind = list(range(len(dt['pfield'])))
# 5. training
step = 0
for epoch in range(args["TRAIN_EPOCH"]): 
    ind = ind_stack[epoch,:]
    if step >= args['MAX_STEP']:
        save_checkpoint(epoch, step, model, optimizer, scheduler, f'{args["save_dir"]}/model', f'step{step:03d}.pth')
        break      
    for i in range(num_batch):      
        step += 1
        if step >= args['MAX_STEP']:
            break  
    ## 1. train dataset
        model.train()
        ind_batch = sorted(ind[i*args["TRAIN_BATCH"]:(i+1)*args["TRAIN_BATCH"]])
        x_batch = torch.tensor(dt['pfield'][ind_batch,...], device=device, dtype=torch.float).unsqueeze(1)
        y_batch = torch.tensor(dt['plabel'][ind_batch,...], device=device, dtype=torch.float).unsqueeze(1)
        m_batch = torch.tensor(dt['pmask' ][ind_batch,...], device=device, dtype=torch.float).unsqueeze(1)
        
        if args['TRAIN_DATA'].split('/')[-1]=='D113':
            pad = torch.zeros((x_batch.shape[0],x_batch.shape[1],x_batch.shape[2],x_batch.shape[3],5), device=device, dtype=torch.float)
            x_batch = torch.cat((pad,x_batch,pad), 4)
            y_batch = torch.cat((pad,y_batch,pad), 4)
            m_batch = torch.cat((pad,m_batch,pad), 4)
        
        x_batch = ( x_batch - dt['input_mean'] ) / dt['input_std']
        y_batch = ( y_batch - dt['label_mean'] ) / dt['label_std']        
        
        pred_batch = model(x_batch)
        l1loss, mdloss, gdloss, loss = total_loss(pred_batch, x_batch, y_batch, m_batch, d, args["TRAIN_W1"], args["TRAIN_W2"]
                                                     ,dt['input_std'], dt['input_mean'], dt['label_std'], dt['label_mean'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        t_loss=loss.item(); t_l1loss=l1loss.item(); t_mdloss=mdloss.item(); t_gdloss=gdloss.item()
        del(x_batch,y_batch,m_batch,pred_batch,loss,l1loss,mdloss,gdloss); torch.cuda.empty_cache();
        logger_train.log({'epoch':epoch+1, 'step':step,
                          't_loss':t_loss, 't_l1loss':t_l1loss, 't_mdloss':t_mdloss, 't_gdloss':t_gdloss,
                          'lr':optimizer.param_groups[0]['lr']})        
        
    save_checkpoint(epoch, step, model, optimizer, scheduler, f'{args["save_dir"]}/model', f'ep{epoch+1:03d}.pth')
    
    ## 2. validation
    model.eval()
    vpred = np.zeros(dt['matrix_size'])
    with torch.no_grad():
        for ii in range(dt['matrix_size'][-1]):
            x_batch = torch.tensor(dt['vfield'][np.newaxis,np.newaxis,...,ii], device=device, dtype=torch.float)
            if args['TRAIN_DATA'].split('/')[-1]=='D113':
                pad = torch.zeros((x_batch.shape[0],x_batch.shape[1],x_batch.shape[2],x_batch.shape[3],5), device=device, dtype=torch.float)
                x_batch = torch.cat((pad,x_batch,pad), 4)
            
            x_batch = ( x_batch - dt['input_mean'] ) / dt['input_std']
            pred_batch = model(x_batch)
            pred = pred_batch[0,0,...].cpu().numpy() * dt['label_std'] + dt['label_mean'] 
            if args['TRAIN_DATA'].split('/')[-1]=='D113':
                pred = pred[...,5:-5]
            vpred[...,ii] = pred * dt['vmask'][...,ii]
            
            l1 = np.mean(np.abs(vpred[...,ii]-dt['vlabel'][...,ii]))
            l2 = np.mean((vpred[...,ii]-dt['vlabel'][...,ii])**2)
            nrmse = compute_nrmse(vpred[...,ii], dt['vlabel'][...,ii], dt['vmask'][...,ii])
            psnr  = compute_psnr( vpred[...,ii], dt['vlabel'][...,ii], dt['vmask'][...,ii])
            logger_val.log({'epoch':epoch+1, 'ori':ii,
                              'l1':l1, 'l2':l2, 'nrmse':nrmse, 'psnr':psnr})                    
    scipy.io.savemat(f'{args["save_dir"]}/val/{epoch+1:03d}.mat', mdict={'pred': vpred})

    dtimg = display_data(); sl=dt['slice'];
    dtimg.figsize = (12,8)
    dtimg.data.append(rotate(dt['vfield'][...,sl,0], -90)); dtimg.v.append((-0.05,0.05)); dtimg.label.append('field')
    dtimg.data.append(rotate(vpred[ ...,sl,0], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('predict')
    dtimg.data.append(rotate(dt['vlabel'][...,sl,0], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('cosmos')
    fig=display_images(1, 3, dtimg, p=False)
    fig.savefig(f'{args["save_dir"]}/val/{epoch+1:03d}.png')          
    print(f"{datetime.now().strftime('%y-%m-%d %H:%M:%S')}   Epoch: {epoch+1:04d}")    