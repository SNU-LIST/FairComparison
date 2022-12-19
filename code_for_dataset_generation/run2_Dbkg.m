%% train2
clear all
load('DLBV/train2/phscos.mat','multimask'); mask=multimask;
load('DPDF/train2/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train2/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train2')
save('Dbkg/train2/multimask.mat','multimask');

%% train3
clear all
load('DLBV/train3/phscos.mat','multimask'); mask=multimask;
load('DPDF/train3/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train3/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train3')
save('Dbkg/train3/multimask.mat','multimask');

%% train4
clear all
load('DLBV/train4/phscos.mat','multimask'); mask=multimask;
load('DPDF/train4/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train4/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train4')
save('Dbkg/train4/multimask.mat','multimask');

%% train5
clear all
load('DLBV/train5/phscos.mat','multimask'); mask=multimask;
load('DPDF/train5/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train5/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train5')
save('Dbkg/train5/multimask.mat','multimask');

%% train6
clear all
load('DLBV/train6/phscos.mat','multimask'); mask=multimask;
load('DPDF/train6/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train6/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train6')
save('Dbkg/train6/multimask.mat','multimask');

%% train7
clear all
load('DLBV/train7/phscos.mat','multimask'); mask=multimask;
load('DPDF/train7/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/train7/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/train7')
save('Dbkg/train7/multimask.mat','multimask');

%% test1
clear all
load('DLBV/test1/phscos.mat','multimask'); mask=multimask;
load('DPDF/test1/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/test1/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/test1')
save('Dbkg/test1/multimask.mat','multimask');

%% test2
clear all
load('DLBV/test2/phscos.mat','multimask'); mask=multimask;
load('DPDF/test2/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/test2/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/test2')
save('Dbkg/test2/multimask.mat','multimask');

%% test3
clear all
load('DLBV/test3/phscos.mat','multimask'); mask=multimask;
load('DPDF/test3/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/test3/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/test3')
save('Dbkg/test3/multimask.mat','multimask');

%% test4
clear all
load('DLBV/test4/phscos.mat','multimask'); mask=multimask;
load('DPDF/test4/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/test4/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/test4')
save('Dbkg/test4/multimask.mat','multimask');

%% test5
clear all
load('DLBV/test5/phscos.mat','multimask'); mask=multimask;
load('DPDF/test5/phscos.mat','multimask'); mask=mask.*multimask;
load('DVSHARP/test5/phscos.mat','multimask'); mask=mask.*multimask;
mkdir('Dbkg/test5')
save('Dbkg/test5/multimask.mat','multimask');