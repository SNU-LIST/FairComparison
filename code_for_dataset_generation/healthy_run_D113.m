load('IMG.mat')
IMG = cat(3,IMG,zeros(size(IMG,1),size(IMG,2),6-rem(size(IMG,3),6),size(IMG,4)));
for t=1:size(IMG,4)
fft_IMG = fft3c(IMG(:,:,:,t));
fft_IMG113 = fft_IMG(:,:,size(IMG,3)/3+1:size(IMG,3)*2/3);
IMG113(:,:,:,t) = ifft3c(fft_IMG113)/3;
end

clear IMG
IMG = IMG113;
save IMG113.mat IMG
spatial_res = [1 1 3];

%--------------------------------------------------------------------------
%% Flip to consistent coordinates as FSLVIEW + BET
%--------------------------------------------------------------------------
for t=1:5
n=num2str(t);
save_nii(make_nii(single(abs(IMG(:,:,:,t))), spatial_res, [], 16), ['mag' n '.nii']);
system(['N4BiasFieldCorrection -d 3 -i mag' n '.nii -o mag' n '_corr.nii']);
system(['bet mag' n '_corr mag' n '_bet -f 0.5']);
system(['gunzip -f *.gz']);
data=load_nii(['mag' n '_bet.nii']);
mask(:,:,:,t)=imerode(imdilate(imerode(data.img~=0,strel('disk',2)),strel('disk',2)),strel('disk',2));
end

for t=1:5
data=load_nii(['mag' num2str(t) '_bet.nii']);
MAG(:,:,:,t)=data.img;
end


%--------------------------------------------------------------------------
%% Background removal
%--------------------------------------------------------------------------
[N(1), N(2), N(3), num_dir] = size(IMG);
pad_size = N / 2;
smv_size = 25;

TE = 0.025;         % sec
B0 = 2.8936;         % Tesla
gyro = 2*pi*42.58;

for t = 1:num_dir
    phase_unwrap = LaplacianPhaseUnwrap(angle(IMG(:,:,:,t)), spatial_res, pad_size);    
    [phase_sharp, mask_sharp] = V_SHARP(phase_unwrap, mask(:,:,:,t), smv_size, pad_size, spatial_res);
    NFM(:,:,:,t) = phase_sharp / (TE*B0*gyro);
    MSK_SHARP(:,:,:,t) = mask_sharp;
end

save NFM.mat MAG NFM MSK_SHARP


%--------------------------------------------------------------------------
%% Compute Flirt transforms
%% Use dof=6 for H vector, dof=12 for registration
%--------------------------------------------------------------------------
cmd = {};
for t=2:5
n=num2str(t);    
cmd = [cmd, ['flirt -in mag' n '_bet -ref mag1_bet -dof 6  -omat rot' n 'to1_6dof']];
cmd = [cmd, ['flirt -in mag' n '_bet -ref mag1_bet -dof 12 -omat rot' n 'to1_12dof']];
end
system_pl(cmd{1},cmd{2},cmd{3},cmd{4},cmd{5},cmd{6},cmd{7},cmd{8});

%--------------------------------------------------------------------------
%% Apply transforms to phase
%--------------------------------------------------------------------------
for t=1:5
n=num2str(t);
save_nii(make_nii(single(NFM(:,:,:,t)), spatial_res, [], 16), ['NFM' n '.nii']);
end

cmd = {};
for t=2:5
n=num2str(t);
cmd = [cmd, ['flirt -in NFM' n ' -ref mag1_bet -applyxfm -init rot' n 'to1_12dof -interp sinc -out NFM' n 'to1']];
end
system_pl(cmd{1},cmd{2},cmd{3},cmd{4});

%%
system('gunzip -f *.gz');

data = load_nii('NFM1.nii');
NFM_REG = data.img;

for t=2:5
data = load_nii(['NFM' num2str(t) 'to1.nii']);
NFM_REG(:,:,:,t) = data.img;
end

msk = prod(single(NFM_REG~=0), 4) ~= 0;     % combined mask
NFM_REG = NFM_REG .* repmat(msk, [1,1,1,5]);

save NFM_REG.mat NFM_REG msk

%--------------------------------------------------------------------------
%% load rotation matrices
%--------------------------------------------------------------------------

R_tot = eye(3);
load rot2to1_6dof;         R_tot(:,:,2) = rot2to1_6dof(1:3,1:3);
load rot3to1_6dof;         R_tot(:,:,3) = rot3to1_6dof(1:3,1:3);
load rot4to1_6dof;         R_tot(:,:,4) = rot4to1_6dof(1:3,1:3);
load rot5to1_6dof;         R_tot(:,:,5) = rot5to1_6dof(1:3,1:3);


for t = 1:size(R_tot,3)
    v = R_tot(:,:,t) * [0;0;1];
    
    disp(['Rotation: ', num2str(acosd(v(3))), ' degrees'])
end




%--------------------------------------------------------------------------
%% create kernels
%--------------------------------------------------------------------------

% 
N = size(NFM_REG(:,:,:,1));
% N = size(phase2);
[ky,kx,kz] = meshgrid(-N(2)/2:N(2)/2-1, -N(1)/2:N(1)/2-1, -N(3)/2:N(3)/2-1);

kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

k2 = kx.^2 + ky.^2 + kz.^2;

kernel = zeross(size(NFM_REG));

for t = 1:size(R_tot,3)
    kernel(:,:,:,t) = fftshift( 1/3 - (kx * R_tot(3,1,t) + ky * R_tot(3,2,t) + kz * R_tot(3,3,t)).^2 ./ (k2 + eps) );    
end



%--------------------------------------------------------------------------
%% Fslview and Flirt are not consistent in AP direction => affects the Back orientation severely
%--------------------------------------------------------------------------

phase_use = fliplr(NFM_REG);
mask_use = fliplr(msk);

%--------------------------------------------------------------------------
%% COSMOS
%--------------------------------------------------------------------------
reg = eps;

Phase_use = zeross(size(phase_use));

for t = 1:size(phase_use,4)
    Phase_use(:,:,:,t) = fftn(phase_use(:,:,:,t));
end

kernel_sum = sum(abs(kernel).^2, 4);
kernel_disp = fftshift(kernel_sum);

chi_cosmos = real( ifftn( sum(kernel .* Phase_use, 4) ./ (reg + kernel_sum) ) ) .* mask_use;
%  chi_cosmos = real( ifftn( sum(kernel .* Phase_use, 4) ./ (reg + kernel_sum) ) );

% imagesc3d2(chi_cosmos - ~chi_cosmos, N/2, 12, [90,90,-90], [-0.1,0.1])

% imagesc3d2(magn_use, N/2, 13, [90,90,-90], [0,0.5])
% 

save cosmos.mat phase_use mask_use chi_cosmos


 %% training_data
%--------------------------------------------------------------------------
NN=size(phase_use);
M2 = [1 0 0 -NN(1)/2; 0 1 0 -NN(2)/2; 0 0 1 -NN(3)/2;0 0 0 1];
M = [R_tot(:, :, 2)', zeros(3, 1);zeros(1, 3) 1];
M3 = inv(M2)*M*M2;
fsltransformation2('regi2.mat', M3)
M = [R_tot(:, :, 3)', zeros(3, 1);zeros(1, 3) 1];
M3 = inv(M2)*M*M2;
fsltransformation2('regi3.mat', M3)
M = [R_tot(:, :, 4)', zeros(3, 1);zeros(1, 3) 1];
M3 = inv(M2)*M*M2;
fsltransformation2('regi4.mat', M3)
M = [R_tot(:, :, 5)', zeros(3, 1);zeros(1, 3) 1];
M3 = inv(M2)*M*M2;
fsltransformation2('regi5.mat', M3)
% R_tot
% load('chi_cosmos.mat')
% load('phase.mat')
%chi_cosmos = chi_cosmos(17:end-16, :, :);
%phase_use = phase_use(17:end-16, :, :, :);

save_nii(make_nii(single(fliplr(phase_use(:,:,:,1))), spatial_res, [], 16), 'phase1.nii');
save_nii(make_nii(single(fliplr(phase_use(:,:,:,2))), spatial_res, [], 16), 'phase2.nii');
save_nii(make_nii(single(fliplr(phase_use(:,:,:,3))), spatial_res, [], 16), 'phase3.nii');
save_nii(make_nii(single(fliplr(phase_use(:,:,:,4))), spatial_res, [], 16), 'phase4.nii');
save_nii(make_nii(single(fliplr(phase_use(:,:,:,5))), spatial_res, [], 16), 'phase5.nii');
save_nii(make_nii(single(fliplr(chi_cosmos)), spatial_res, [], 16), 'cosmos.nii');
save_nii(make_nii(single(fliplr(mask_use)), spatial_res, [], 16), 'mask.nii');

cmd = {};
for t=2:5
n=num2str(t);
cmd = [cmd, ['flirt -in phase' n ' -ref phase' n ' -applyxfm -init regi' n '.mat  -interp sinc -sincwidth 15 -sincwindow hanning -out phase' n '_out']];
cmd = [cmd, ['flirt -in cosmos  -ref cosmos -applyxfm -init regi' n '.mat  -interp sinc -sincwidth 15 -sincwindow hanning -out cosmos' n]];
cmd = [cmd, ['flirt -in mask    -ref mask   -applyxfm -init regi' n '.mat  -interp sinc -sincwidth 15 -sincwindow hanning -out mask' n]];
end
system_pl(cmd{1},cmd{2},cmd{3},cmd{4},cmd{5},cmd{6},cmd{7},cmd{8},cmd{9},cmd{10},cmd{11},cmd{12});
system('gunzip -f *.gz');

%%
multiphs = phase_use(:, :, :, 1);
data = load_nii('phase2_out.nii');
multiphs(:,:,:,2) = fliplr(data.img);
data = load_nii('phase3_out.nii');
multiphs(:,:,:,3) = fliplr(data.img);
data = load_nii('phase4_out.nii');
multiphs(:,:,:,4) = fliplr(data.img);
data = load_nii('phase5_out.nii');
multiphs(:,:,:,5) = fliplr(data.img);

multicos = chi_cosmos;
data = load_nii('cosmos2.nii');
multicos(:,:,:,2) = fliplr(data.img);
data = load_nii('cosmos3.nii');
multicos(:,:,:,3) = fliplr(data.img);
data = load_nii('cosmos4.nii');
multicos(:,:,:,4) = fliplr(data.img);
data = load_nii('cosmos5.nii');
multicos(:,:,:,5) = fliplr(data.img);

multimask = single(mask_use);
data = load_nii('mask2.nii');
multimask(:,:,:,2) = fliplr(data.img);
data = load_nii('mask3.nii');
multimask(:,:,:,3) = fliplr(data.img);
data = load_nii('mask4.nii');
multimask(:,:,:,4) = fliplr(data.img);
data = load_nii('mask5.nii');
multimask(:,:,:,5) = fliplr(data.img);

save phscos.mat multimask multicos multiphs