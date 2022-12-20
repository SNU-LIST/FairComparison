import scipy.io
import numpy as np
import h5py
import time
import os
import sys

PS = 64
patch_num = [6,8,7]
print(f'pwd : {os.getcwd()}')
result_file = h5py.File(f'../data/D111/training_data_patch.hdf5', 'w')

# Patch the input & mask file ----------------------------------------------------------------
print("####patching input####")
patches_field = []
patches_mask = []
for s in [2,3,4,5,6]:
    m = scipy.io.loadmat(f'../data/D111/result/train{s}.mat')
    field = m['multiphs']
    mask  = m['multimask']
    matrix_size = np.shape(mask)
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]; 
    print(strides)
    for idx in range(matrix_size[-1]):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches_field.append(field[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   idx])              
                    patches_mask.append(mask[
                                        i * strides[0]:i * strides[0] + PS,
                                        j * strides[1]:j * strides[1] + PS,
                                        k * strides[2]:k * strides[2] + PS,
                                        idx])
print("Done!")

patches_field = np.array(patches_field, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='bool', copy=False)
print("Final input data size : " + str(np.shape(patches_field)))

input_mean = np.mean(patches_field[patches_mask > 0])
input_std  = np.std( patches_field[patches_mask > 0])

result_file.create_dataset('pfield', data=patches_field)
result_file.create_dataset('pmask', data=patches_mask)
del patches_field


patches_label = []
for s in [2,3,4,5,6]:
    m = scipy.io.loadmat(f'../data/D111/result/train{s}.mat')
    label = m['multicos']
    mask  = m['multimask']
    matrix_size = np.shape(mask)
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]; 
    for idx in range(matrix_size[-1]):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches_label.append(label[
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   idx])        
print("Done!")

patches_label = np.array(patches_label, dtype='float32', copy=False)
print("Final label data size : " + str(np.shape(patches_label)))

label_mean = np.mean(patches_label[patches_mask > 0])
label_std  = np.std( patches_label[patches_mask > 0])
n_element = np.sum(patches_mask)

result_file.create_dataset('plabel', data=patches_label)


del patches_label
del patches_mask
result_file.close()

scipy.io.savemat(f'../data/D111/training_data_patch_norm_factor.mat',
                 mdict={'input_mean': input_mean, 'input_std': input_std,
                        'label_mean': label_mean, 'label_std': label_std, 'n_element': n_element})

