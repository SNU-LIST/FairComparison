% In each folder (e.g. Data/D111/test1), IMG.mat is required

clear all
cd test1
system('cp -rv ../../D111/test1/mag*.nii ./')
system('cp -rv ../../D111/test1/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd test2
system('cp -rv ../../D111/test2/mag*.nii ./')
system('cp -rv ../../D111/test2/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd test3
system('cp -rv ../../D111/test3/mag*.nii ./')
system('cp -rv ../../D111/test3/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd test4
system('cp -rv ../../D111/test4/mag*.nii ./')
system('cp -rv ../../D111/test4/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd test5
system('cp -rv ../../D111/test5/mag*.nii ./')
system('cp -rv ../../D111/test5/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train2
system('cp -rv ../../D111/train2/mag*.nii ./')
system('cp -rv ../../D111/train2/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train3
system('cp -rv ../../D111/train3/mag*.nii ./')
system('cp -rv ../../D111/train3/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train4
system('cp -rv ../../D111/train4/mag*.nii ./')
system('cp -rv ../../D111/train4/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train5
system('cp -rv ../../D111/train5/mag*.nii ./')
system('cp -rv ../../D111/train5/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train6
system('cp -rv ../../D111/train6/mag*.nii ./')
system('cp -rv ../../D111/train6/rot* ./')
healthy_run_DVSHARP
cd ..

clear all
cd train7
system('cp -rv ../../D111/train7/mag*.nii ./')
system('cp -rv ../../D111/train7/rot* ./')
healthy_run_DVSHARP
cd ..