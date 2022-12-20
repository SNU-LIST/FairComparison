%% train2
clear all
load('train2/phscos.mat');

idx=[25,35,9;25,35,9;25,35,9;25,35,1;25,40,9;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
%figure, imshow_4d(cat(1,multicos2,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train2/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% train3
clear all
load('train3/phscos.mat');

idx=[25,25,9;25,25,9;25,25,9;25,35,1;25,30,9;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos3,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train3/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

%% train4
clear all
% load('eunjung_data/train_subject4/phscos.mat','multicos4','mask4');
load('train4/phscos.mat');
multicos=flip(flip(multicos,1),3); multiphs=flip(flip(multiphs,1),3); multimask=flip(flip(multimask,1),3);

idx=[37,28,9;37,28,9;42,25,9;37,35,5;37,30,9;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos4,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train4/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

%% train5
clear all
% load('eunjung_data/train_subject5/phscos.mat','multicos5','mask5');
load('train5/phscos.mat');

idx=[37,28,9;37,28,9;42,25,9;37,35,5;37,30,9;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos5,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train5/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% train6
clear all
% load('eunjung_data/train_subject6/phscos.mat','multicos6','mask6');
load('train6/phscos.mat'); 

idx=[37,28,9;37,28,9;42,25,3;37,35,13;37,30,2;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos6,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train6/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% train7
clear all
% load('eunjung_data/train_subject7/phscos.mat','multicos7','mask7');
load('train7/phscos.mat'); 

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos7,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('train7/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');


%% test1
clear all
% load('eunjung_data/test_subject1/phscos.mat','multicos8','mask8');
load('test1/phscos.mat'); 

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos8,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test1/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% test2
clear all
% load('eunjung_data/test_subject2/phscos.mat','multicos9','mask9');
load('test2/phscos.mat'); 
multicos=flip(multicos,2); multiphs=flip(multiphs,2); multimask=flip(multimask,2);

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos9,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test2/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% test3
clear all
% load('eunjung_data/test_subject3/phscos.mat','multicos10','mask10');
load('test3/phscos.mat'); 

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos10,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test3/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% test4
clear all
% load('eunjung_data/test_subject4/phscos.mat','multicos11','mask11');
load('test4/phscos.mat'); 

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos11,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test4/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);


%% test5
clear all
% load('eunjung_data/test_subject5/phscos.mat','multicos12','mask12');
load('test5/phscos.mat'); 

idx=[37,20,1;37,20,1;42,20,1;37,25,1;37,20,1;];
for i=1:5
multicos_(:,:,:,i)=multicos(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multiphs_(:,:,:,i)=multiphs(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i);
multimask_(:,:,:,i)=multimask(idx(i,1):176-1+idx(i,1),idx(i,2):176-1+idx(i,2),idx(i,3):160-1+idx(i,3),i)>0.7;

mask_crop(:,:,:,i)=multimask_(:,:,:,i); mask_crop([1,end],:,:,:)=10; mask_crop(:,[1,end],:,:)=10;
end
% figure, imshow_4d(cat(1,multicos12,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
% figure, imshow_4d(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);

multicos=multicos_; multiphs=multiphs_; multimask=multimask_;
save('test5/phscos_crop.mat', 'multicos', 'multiphs', 'multimask');
% figure, imshow_3df(cat(1,multicos_,3*multiphs_,mask_crop),[-0.15,0.15]);
