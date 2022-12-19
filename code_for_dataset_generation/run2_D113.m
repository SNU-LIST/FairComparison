%% train2
clear all
load('train2/phscos.mat');

idx=[25,35,1;25,35,1;45,35,1;25,45,1;25,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train2/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train3
clear all
load('train3/phscos.mat');

idx=[25,35,1;25,35,1;45,35,1;25,45,1;25,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train3/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train4
clear all
load('train4/phscos.mat');
multicos=flip(flip(multicos,1),3); multiphs=flip(flip(multiphs,1),3); multimask=flip(flip(multimask,1),3);

idx=[40,25,4;25,30,4;65,25,4;35,20,4;35,40,4;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train4/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train5
clear all
load('train5/phscos.mat');

idx=[40,25,4;65,30,4;25,25,4;45,40,4;45,10,4;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train5/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train6
clear all
load('train6/phscos.mat');

idx=[40,25,4;25,30,4;65,35,1;45,10,4;45,45,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train6/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train7
clear all
load('train7/phscos.mat');

idx=[40,15,1;25,20,1;45,15,1;45,40,1;45,5,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train7/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test1
clear all
load('test1/phscos.mat');

idx=[40,20,1;25,20,1;55,25,1;45,10,1;45,35,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test1/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test2
clear all
load('test2/phscos.mat');
multicos=flip(multicos,2); multiphs=flip(multiphs,2); multimask=flip(multimask,2);

idx=[40,20,1;25,20,1;55,25,1;45,45,1;45,15,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test2/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test3
clear all
load('test3/phscos.mat');

idx=[40,20,1;25,20,1;55,25,1;45,15,1;45,45,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test3/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test4
clear all
load('test4/phscos.mat');

idx=[40,20,1;45,20,1;25,25,1;45,15,1;45,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test4/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test5
clear all
load('test5/phscos.mat');

idx=[40,20,1;45,20,1;25,25,1;45,25,1;45,10,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):54-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%%figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test5/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');