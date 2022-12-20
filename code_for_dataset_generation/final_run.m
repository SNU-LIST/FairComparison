addpath(genpath('.'))
addpath(genpath('util'))

%% For generation D111,
cd ../data/D111
run1_D111
run2_D111
run3_D111

%% For generation D113,
cd ../D113
run1_D113
run2_D113
run3_D113

%% For generation DLBV, DPDF, and DVSHARP
cd ../DLBV
run1_DLBV

cd ../DPDF
run1_DPDF

cd ../DVSHARP
run1_DVSHARP

cd ..
run2_Dbkg

cd DLBV
run2_DLBV

cd ../DPDF
run2_DLBV

cd ../DVSHARP
run2_DLBV

cd ../DLBV
run3_D111

cd ../DPDF
run3_D111

cd ../DVSHARP
run3_D111
