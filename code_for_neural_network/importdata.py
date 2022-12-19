import scipy.io
import h5py
import nibabel as nib
import numpy as np
import glob, os

def load_traindata(path, load_scale=False):
    name = path.split('/')[-1]
    if name in ['D','D111','DVSHARP', 'DRESHARP', 'DLBV', 'DPDF', 'D113', 'Dshanghai', 'D111-4', 'DVSHARP2', 'DRESHARP2', 'DLBV2', 'DPDF2', 'DiHARP2']:       
        
        m = scipy.io.loadmat(f'/home/chungseok/project/compareqsmnet/data/{name}/training_data_patch_norm_factor.mat')
        input_mean = m['input_mean'].item()
        label_mean = m['label_mean'].item()
        input_std = m['input_std'].item()
        label_std = m['label_std'].item()
        n_element = m['n_element']
        if load_scale:
            print(f'load_scale : {name}')
            return {'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}
        
        h = h5py.File(f'{path}/training_data_patch.hdf5','r')
        pfield = h['pfield']
        plabel = h['plabel']
        pmask  = h['pmask' ]

        if name in ['D111-4']:
            pfield = pfield[:len(pfield)*0.8,...]
            plabel = plabel[:len(pfield)*0.8,...]
            pmask  = pmask[ :len(pfield)*0.8,...]

        if name in ['Dshanghai']:
            val=scipy.io.loadmat(f'/home/chungseok/project/compareqsmnet/data/{name}/train5.mat')
        else:
            val=scipy.io.loadmat(f'/home/chungseok/project/compareqsmnet/data/{name}/train7.mat')
        vfield=val['multiphs']
        vlabel=val['multicos']
        vmask =val['multimask'].astype(bool)

        if name in ['D','D111','DVSHARP', 'DRESHARP', 'DLBV', 'DPDF','Dshanghai', 'DVSHARP2', 'DRESHARP2', 'DLBV2', 'DPDF2', 'DiHARP2']:
            print(f'load_traindata : {name}')
            return {'pfield':pfield,'plabel':plabel,'pmask':pmask,'patch_size':pmask.shape,'voxel_size':(1,1,1),
                    'vfield':vfield,'vlabel':vlabel,'vmask':vmask,'matrix_size':vmask.shape,'slice':73,
                    'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}

        elif name in ['D113']:
            print(f'load_traindata : {name}')
            return {'pfield':pfield,'plabel':plabel,'pmask':pmask,'patch_size':pmask.shape,'voxel_size':(1,1,3),
                    'vfield':vfield,'vlabel':vlabel,'vmask':vmask,'matrix_size':vmask.shape,'slice':25,
                    'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}    

        
def load_testdata(path):
    name = path.split('/')[-1]
    if name in ['D','D111','DVSHARP', 'DRESHARP', 'DLBV', 'DPDF', 'D113', 'DVSHARP2', 'DRESHARP2', 'DLBV2', 'DPDF2', 'DiHARP2']:           
        tfield = []; tlabel = []; tmask = []; 
            
        for sub in ['test1', 'test2', 'test3', 'test4', 'test5']:
            m = scipy.io.loadmat(f'{path}/{sub}.mat')
            tfield.append(m['multiphs'])
            tlabel.append(m['multicos'])
            tmask.append(m['multimask'].astype(bool))
            
        if name in ['D','D111','DVSHARP', 'DRESHARP', 'DLBV', 'DPDF', 'DVSHARP2', 'DRESHARP2', 'DLBV2', 'DPDF2', 'DiHARP2']:
            print(f'load_testdata : {name}')
            return {'num':5,'tfield':tfield,'tlabel':tlabel,'tmask':tmask,'matrix_size':tmask[0].shape,'slice':73}

        elif name in ['D113']:
            print(f'load_testdata : {name}')
            return {'num':5,'tfield':tfield,'tlabel':tlabel,'tmask':tmask,'matrix_size':tmask[0].shape,'slice':25}
    
    elif name in ['Dshanghai']:
        tfield = []; tlabel = []; tmask = []; 
            
        for sub in ['test6', 'test7', 'test8']:
            m = scipy.io.loadmat(f'{path}/{sub}.mat')
            tfield.append(m['multiphs'])
            tlabel.append(m['multicos'])
            tmask.append(m['multimask'].astype(bool))        
        print(f'load_testdata : {name}')
        return {'num':3,'tfield':tfield,'tlabel':tlabel,'tmask':tmask,'matrix_size':tmask[0].shape,'slice':73}            
            