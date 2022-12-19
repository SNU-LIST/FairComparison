%--------------------------------------------------------------------------
%% load data
%--------------------------------------------------------------------------

% change to your data directory
% data_path = '/home/svr/Desktop/list2/JaeYeonYoon/COSMOSsub1/rotN/';
data_path = '/data/trset/cosmos3/cosmosdata/COSMOSsub1/test/';

dt = mapVBVD([data_path, 'meas_MID831_gre_TILT1_FID611.dat'], 'removeOS');
% recon_path = '/home/svr/Documents/MATLAB/coil/';
recon_path = '/home/svr/Downloads/coil/';
% dt = mapVBVD([data_path, fname], 'removeOS');




%--------------------------------------------------------------------------
%% GCC coil compression: estimate compression from patref 
%--------------------------------------------------------------------------

% readout size
res = dt.image(:,1,1,1,1);
num_ro = length(res)


% separate acs
data = dt.refscan();

data = permute(data, [1,3,4,2]);       

 
% zero-pad in readout
data = padarray(data, [(num_ro - size(data,1)), 0, 0, 0] / 2);



num_chan = 16;      % set this to desired coil count
dim = 1;            % readout axis


% compute gcc compression matrices
gccmtx = calcGCCMtx(data, dim);

gccmtx_aligned = alignCCMtx(gccmtx(:,1:num_chan,:));
 
CCDATA_aligned = CC( data, dim, gccmtx_aligned);

     
% compressed data
IMG_gcc = ifft3c(CCDATA_aligned);


% full data
IMG = ifft3c( data );    


rmse_gcc = 100 * norm2(rsos(IMG_gcc, 4) - rsos(IMG, 4)) / norm2( rsos(IMG, 4) )


mosaic( rsos(IMG, 4), 5, 6, 1, '', genCaxis(rsos(IMG, 4)), 90 ), axis square
mosaic( rsos(IMG_gcc, 4), 5, 6, 2, num2str(rmse_gcc), genCaxis(rsos(IMG, 4)), 90 ), axis square



clear IMG IMG_gcc

 

 
%--------------------------------------------------------------------------
%% apply coil compression to patref and data 
%--------------------------------------------------------------------------

dat = squeeze(dt.image());
dat = permute(dat, [1,3,4,2,5]);


ref = squeeze(dt.refscan());
ref = permute(ref, [1,3,4,2,5]);


ref = padarray(ref, [(num_ro - size(ref,1)), 0, 0, 0] / 2);

ref_gcc = CC( ref, dim, gccmtx_aligned);
dat_gcc = CC( dat, dim, gccmtx_aligned);


ref_gcc = single(ref_gcc);
dat_gcc = single(dat_gcc);


rmse_dat = 100 * norm2(rsos(ifft3c(dat_gcc), 4) - rsos(ifft3c(dat), 4)) / norm2( rsos(ifft3c(dat), 4) )
rmse_ref = 100 * norm2(rsos(ifft3c(ref_gcc), 4) - rsos(ifft3c(ref), 4)) / norm2( rsos(ifft3c(ref), 4) )


clear ref dat

 

% pad due to accl
dat_gcc = padarray(dat_gcc, [0,1,0,0,0]);
dat_gcc = padarray(dat_gcc, [0,0,1,0,0], 'post');
   


%--------------------------------------------------------------------------
%% ifft in readout
%--------------------------------------------------------------------------
 
 
% x,ky,kz,chan,
REF = fftshift(ifft(fftshift( ref_gcc, 1 ), [], 1), 1);
DAT = fftshift(ifft(fftshift( dat_gcc, 1 ), [], 1), 1);




%--------------------------------------------------------------------------
%% recon single readout slice and coil combine
%--------------------------------------------------------------------------

lambda_tik = eps;


num_acs = [30,30];          % size reduced due to 1 voxel circshift
kernel_size = [3,3];        % odd kernel size


num_fa = 1;    
num_eco = 1;


Rz = 2;
Ry = 2;


N = [size(DAT,2), size(DAT,3)];
num_chan = size(DAT,4);

del_z = zeros(num_chan,1);
del_y = zeros(num_chan,1);

IMG = zeross( [num_ro, N, num_eco, num_fa] );


for slice_select = 1:num_ro

    disp(['slice: ', num2str(slice_select)])
    
    Kspace_Sampled = squeeze( DAT(slice_select, :,:,:,:,:) );
    Kspace_Acs = squeeze( REF(slice_select, :,:,:,:,:) );


    size_kspace = size(Kspace_Sampled(:,:,1,1,1));
    size_ref = size(Kspace_Acs(:,:,1,1,1));


    Kspace_Acs = padarray( Kspace_Acs, [size_kspace-size_ref, 0, 0, 0]/2 );

    Img_Grappa = zeross([N, num_chan, num_eco, num_fa]);
    tic
    for f = 1:num_fa
        kspace_acs = Kspace_Acs(:,:,:,f);

        for t = 1:num_eco
            kspace_sampled = Kspace_Sampled(:,:,:,t,f);

            Img_Grappa(:,:,:,t,f) = grappa_gfactor_2d_jvc2( kspace_sampled, kspace_acs, Rz, Ry, num_acs, kernel_size, lambda_tik, 0, del_z, del_y );
        end
    end
    toc

    sens = CoilSense_ESPIRIT2d( permute( ifft2c( Kspace_Acs(:,:,:,1) ), [1,2,4,3] ), 0, num_acs(1), recon_path);

    sens = repmat(permute(sens, [1,2,4,3]), [1,1,1,num_eco,num_fa]);

    Img_Combo = sum(Img_Grappa .* conj(sens), 3) ./ (eps + sum(abs(sens).^2, 3));

    IMG(slice_select,:,:,:,:) = Img_Combo;
    
end


 
% mosaic( squeeze(IMG(slice_select,:,:)), 1, 1, 3, '', [0,6e-5], -90 )
% mosaic( angle(squeeze(IMG(slice_select,:,:))), 1, 1, 4, '', [-pi,pi], -90 )


% imagesc3d2(IMG, size(IMG)/2, 1, [-90,-90,90], [0,6e-5])
% imagesc3d2(angle(IMG), size(IMG)/2, 2, [-90,-90,90], [-pi,pi])

save test.mat IMG

 

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% BELOW IS FOR COSMOS PROCESSING
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------




%--------------------------------------------------------------------------
%% Flip to consistent coordinates as FSLVIEW
%--------------------------------------------------------------------------
% load('IMG_R_orig.mat')
% data = load_nii('EUNN_20180128_001_004_gre_TILT4.nii');
% temp = single(flipud(data.img));
% Img_F_dicom = temp;
% IMG = IMG(:,:,10:169);
% tt = zeros(size(IMG));
% tt(:,1:end-1,:) = IMG(:,2:end,:);
% figure;imshow3d(abs(temp(:,:,104)))
% figure;imshow3d(abs(tt(:,:,104)))
% IMG = tt;
% save IMG_F.mat IMG Img_F_dicom

spatial_res = [1,1,1];
hf = 0.5;
hfilt2 = tukeywin(256, hf)*tukeywin(224, hf)';
hfilt3 = repmat(hfilt2, 1, 1, 160).*permute(repmat(tukeywin(160, hf), 1, 256, 224), [2 3 1]);

% neutral pos
% load /home/svr/Desktop/list2/JaeYeonYoon/COSMOSsub2/rotN/IMG_N
load /data/trset/cosmos3/cosmosdata/COSMOSsub4/rotN/IMG_N
% load /data/trset/cosmos/subj3/rotN/IMG_N

IMG = ifft3c(fft3c(IMG3).*hfilt3);
Img_N = imrotate(flipdim(flipdim(IMG, 2), 3), 180);   
Img_N_dicom = imrotate(flipdim(flipdim(Img_N_dicom, 2), 3), 180);

save_nii(make_nii(single(abs(Img_N)), spatial_res, [], 16), 'Abs_N.nii')
save_nii(make_nii(single(abs(Img_N_dicom)), spatial_res, [], 16), 'Abs_N_dicom.nii')



% left pos
load /data/trset/cosmos3/cosmosdata/COSMOSsub4/rotL/IMG_L
% load /data/trset/cosmos/subj3/rotL/IMG_L

IMG = ifft3c(fft3c(IMG3).*hfilt3);
Img_L = imrotate(flipdim(flipdim(IMG, 2), 3), 180);   
Img_L_dicom = imrotate(flipdim(flipdim(Img_L_dicom, 2), 3), 180);

save_nii(make_nii(single(abs(Img_L)), spatial_res, [], 16), 'Abs_L.nii')
save_nii(make_nii(single(abs(Img_L_dicom)), spatial_res, [], 16), 'Abs_L_dicom.nii')



% right pos
load /data/trset/cosmos3/cosmosdata/COSMOSsub4/rotR/IMG_R
% load /data/trset/cosmos/subj3/rotR/IMG_R

IMG = ifft3c(fft3c(IMG3).*hfilt3);
Img_R = imrotate(flipdim(flipdim(IMG, 2), 3), 180);           
Img_R_dicom = imrotate(flipdim(flipdim(Img_R_dicom, 2), 3), 180);


save_nii(make_nii(single(abs(Img_R)), spatial_res, [], 16), 'Abs_R.nii')
save_nii(make_nii(single(abs(Img_R_dicom)), spatial_res, [], 16), 'Abs_R_dicom.nii')




% back pos
load /data/trset/cosmos3/cosmosdata/COSMOSsub4/rotB/IMG_B
% load /data/trset/cosmos/subj3/rotB/IMG_B

IMG = ifft3c(fft3c(IMG3).*hfilt3);
Img_B = imrotate(flipdim(flipdim(IMG, 2), 3), 180);           
Img_B_dicom = imrotate(flipdim(flipdim(Img_B_dicom, 2), 3), 180);


save_nii(make_nii(single(abs(Img_B)), spatial_res, [], 16), 'Abs_B.nii')
save_nii(make_nii(single(abs(Img_B_dicom)), spatial_res, [], 16), 'Abs_B_dicom.nii')




% front pos
load /data/trset/cosmos3/cosmosdata/COSMOSsub4/rotF/IMG_F
% load /data/trset/cosmos/subj3/rotF/IMG_F

IMG = ifft3c(fft3c(IMG3).*hfilt3);
Img_F = imrotate(flipdim(flipdim(IMG, 2), 3), 180);           
Img_F_dicom = imrotate(flipdim(flipdim(Img_F_dicom, 2), 3), 180);


save_nii(make_nii(single(abs(Img_F)), spatial_res, [], 16), 'Abs_F.nii')
save_nii(make_nii(single(abs(Img_F_dicom)), spatial_res, [], 16), 'Abs_F_dicom.nii')




%--------------------------------------------------------------------------
%% BET masking
%--------------------------------------------------------------------------
% 
unix('bet Abs_N Abs_N_bet  -f 0.4')
unix('bet Abs_L_dicom Abs_L_bet  -f 0.5')
unix('bet Abs_R_dicom Abs_R_bet  -f 0.5')
unix('bet Abs_B_dicom Abs_B_bet  -f 0.5')
unix('bet Abs_F_dicom Abs_F_bet  -f 0.5')


unix('gunzip *.gz')


data = load_nii('Abs_N_bet.nii');
MSK = data.img ~= 0;
mag_notrot = data.img;

data = load_nii('Abs_L_bet.nii');
MSK(:,:,:,2) = data.img ~= 0;
mag_notrot(:,:,:,2) = data.img;

data = load_nii('Abs_R_bet.nii');
MSK(:,:,:,3) = data.img ~= 0;
mag_notrot(:,:,:,3) = data.img;

data = load_nii('Abs_B_bet.nii');
MSK(:,:,:,4) = data.img ~= 0;
mag_notrot(:,:,:,4) = data.img;

data = load_nii('Abs_F_bet.nii');
MSK(:,:,:,5) = data.img ~= 0;
mag_notrot(:,:,:,5) = data.img;

int_norm



save_nii(make_nii(single(mag_notrot_norm2(:,:,:,1)), spatial_res, [], 16), 'mag_N.nii')
save_nii(make_nii(single(mag_notrot_norm2(:,:,:,2)), spatial_res, [], 16), 'mag_L.nii')
save_nii(make_nii(single(mag_notrot_norm2(:,:,:,3)), spatial_res, [], 16), 'mag_R.nii')
save_nii(make_nii(single(mag_notrot_norm2(:,:,:,4)), spatial_res, [], 16), 'mag_B.nii')
save_nii(make_nii(single(mag_notrot_norm2(:,:,:,5)), spatial_res, [], 16), 'mag_F.nii')

IMG = cat(4, Img_N, Img_L, Img_R, Img_B, Img_F) .* MSK;
PHS = angle(IMG);
mag_notrot = abs(IMG);


for t = 1:size(PHS,4)
    imagesc3d2(IMG(:,:,:,t), size(IMG)/2, t, [90,90,90], [0,2e-5])
end




%--------------------------------------------------------------------------
%% unwrap and remove background phase
%--------------------------------------------------------------------------


prot = read_meas_prot([data_path, 'meas_MID00228_FID31431_gre_N.dat'])


[N(1), N(2), N(3), num_dir] = size(IMG)

pad_size = N / 2;

smv_size = 25;


TE = prot.alTE * 1e-6;         % sec
B0 = prot.flNominalB0;         % Tesla
gyro = 2*pi*42.58;

NFM = zeross(size(PHS));
MSK_SHARP = zeross(size(PHS));


for t = 1:num_dir
    phase_unwrap = LaplacianPhaseUnwrap(PHS(:,:,:,t), spatial_res, pad_size);
    
    [phase_sharp, mask_sharp] = V_SHARP(phase_unwrap,  MSK(:,:,:,t), smv_size, pad_size, spatial_res);
    
    NFM(:,:,:,t) = phase_sharp / (TE*B0*gyro);
    
    MSK_SHARP(:,:,:,t) = mask_sharp;

    imagesc3d2(NFM(:,:,:,t), N/2, t, [90,90,90], [-0.05,0.05])
end





%--------------------------------------------------------------------------
%% Compute Flirt transforms
%% Use dof=6 for H vector, dof=12 for registration
%--------------------------------------------------------------------------


unix('flirt -in Abs_L_bet  -ref Abs_N_bet -dof 6  -omat L2N_6dof')
unix('flirt -in Abs_L_bet  -ref Abs_N_bet -dof 12 -omat L2N_12dof')

unix('flirt -in Abs_R_bet  -ref Abs_N_bet -dof 6  -omat R2N_6dof')
unix('flirt -in Abs_R_bet  -ref Abs_N_bet -dof 12 -omat R2N_12dof')

unix('flirt -in Abs_B_bet  -ref Abs_N_bet -dof 6  -omat B2N_6dof')
unix('flirt -in Abs_B_bet  -ref Abs_N_bet -dof 12 -omat B2N_12dof')
 
unix('flirt -in Abs_F_bet  -ref Abs_N_bet -dof 6  -omat F2N_6dof')
unix('flirt -in Abs_F_bet  -ref Abs_N_bet -dof 12 -omat F2N_12dof')



unix('flirt -in Abs_L_bet  -ref Abs_N_bet -applyxfm -init L2N_12dof  -interp sinc -out Abs_L2N_bet')
unix('flirt -in Abs_R_bet  -ref Abs_N_bet -applyxfm -init R2N_12dof  -interp sinc -out Abs_R2N_bet')
unix('flirt -in Abs_B_bet  -ref Abs_N_bet -applyxfm -init B2N_12dof  -interp sinc -out Abs_B2N_bet')
unix('flirt -in Abs_F_bet  -ref Abs_N_bet -applyxfm -init F2N_12dof  -interp sinc -out Abs_F2N_bet')

%% Use dof=6 for H vector, dof=12 for registration
%--------------------------------------------------------------------------


unix('flirt -in mag_L  -ref mag_N -dof 6  -omat L2N_6dof')
unix('flirt -in mag_L  -ref mag_N -dof 12 -omat L2N_12dof')

unix('flirt -in mag_R  -ref mag_N -dof 6  -omat R2N_6dof')
unix('flirt -in mag_R  -ref mag_N -dof 12 -omat R2N_12dof')

unix('flirt -in mag_B  -ref mag_N -dof 6  -omat B2N_6dof')
unix('flirt -in mag_B  -ref mag_N -dof 12 -omat B2N_12dof')
 
unix('flirt -in mag_F  -ref mag_N -dof 6  -omat F2N_6dof')
unix('flirt -in mag_F  -ref mag_N -dof 12 -omat F2N_12dof')



unix('flirt -in mag_L  -ref mag_N -applyxfm -init L2N_12dof  -interp sinc -out Abs_L2N_bet')
unix('flirt -in mag_R  -ref mag_N -applyxfm -init R2N_12dof  -interp sinc -out Abs_R2N_bet')
unix('flirt -in mag_B  -ref mag_N -applyxfm -init B2N_12dof  -interp sinc -out Abs_B2N_bet')
unix('flirt -in mag_F  -ref mag_N -applyxfm -init F2N_12dof  -interp sinc -out Abs_F2N_bet')


%--------------------------------------------------------------------------
%% Apply transforms to phase
%--------------------------------------------------------------------------

save_nii(make_nii(single(NFM(:,:,:,1)), spatial_res, [], 16), 'NFM_N.nii')
save_nii(make_nii(single(NFM(:,:,:,2)), spatial_res, [], 16), 'NFM_L.nii')
save_nii(make_nii(single(NFM(:,:,:,3)), spatial_res, [], 16), 'NFM_R.nii')
save_nii(make_nii(single(NFM(:,:,:,4)), spatial_res, [], 16), 'NFM_B.nii')
save_nii(make_nii(single(NFM(:,:,:,5)), spatial_res, [], 16), 'NFM_F.nii')



% unix('flirt -in NFM_L  -ref Abs_N_bet -applyxfm -init L2N_12dof  -interp sinc -out NFM_L2N')
% unix('flirt -in NFM_R  -ref Abs_N_bet -applyxfm -init R2N_12dof  -interp sinc -out NFM_R2N')
% unix('flirt -in NFM_B  -ref Abs_N_bet -applyxfm -init B2N_12dof  -interp sinc -out NFM_B2N')
% unix('flirt -in NFM_F  -ref Abs_N_bet -applyxfm -init F2N_12dof  -interp sinc -out NFM_F2N')

unix('flirt -in NFM_L  -ref mag_N -applyxfm -init L2N_12dof  -interp sinc -out NFM_L2N')
unix('flirt -in NFM_R  -ref mag_N -applyxfm -init R2N_12dof  -interp sinc -out NFM_R2N')
unix('flirt -in NFM_B  -ref mag_N -applyxfm -init B2N_12dof  -interp sinc -out NFM_B2N')
unix('flirt -in NFM_F  -ref mag_N -applyxfm -init F2N_12dof  -interp sinc -out NFM_F2N')



unix('gunzip *.gz')

data = load_nii('NFM_N.nii');
NFM_REG = data.img;

data = load_nii('NFM_L2N.nii');
NFM_REG(:,:,:,2) = data.img;

data = load_nii('NFM_R2N.nii');
NFM_REG(:,:,:,3) = data.img;

data = load_nii('NFM_B2N.nii');
NFM_REG(:,:,:,4) = data.img;

data = load_nii('NFM_F2N.nii');
NFM_REG(:,:,:,5) = data.img;

data = load_nii('mag_N.nii');
magn_use = data.img;

data = load_nii('Abs_L2N_bet.nii');
magn_use(:,:,:,2) = data.img;

data = load_nii('Abs_R2N_bet.nii');
magn_use(:,:,:,3) = data.img;

data = load_nii('Abs_B2N_bet.nii');
magn_use(:,:,:,4) = data.img;

data = load_nii('Abs_F2N_bet.nii');
magn_use(:,:,:,5) = data.img;



msk = prod(single(NFM_REG~=0), 4) ~= 0;     % combined mask


NFM_REG = NFM_REG .* repmat(msk, [1,1,1,num_dir]);

for t = 1:num_dir
    imagesc3d2(NFM_REG(:,:,:,t), N/2, t, [90,90,90], [-0.05,0.05])
end





%--------------------------------------------------------------------------
%% load rotation matrices
%--------------------------------------------------------------------------

R_tot = eye(3);
load L2N_6dof;         R_tot(:,:,2) = L2N_6dof(1:3,1:3);
load R2N_6dof;         R_tot(:,:,3) = R2N_6dof(1:3,1:3);
load B2N_6dof;         R_tot(:,:,4) = B2N_6dof(1:3,1:3);
load F2N_6dof;         R_tot(:,:,5) = F2N_6dof(1:3,1:3);


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

    mosaic(fftshift(kernel(:,:,:,t)), 12, 15, t, '', [-2/3,1/3])
end



%--------------------------------------------------------------------------
%% Fslview and Flirt are not consistent in AP direction => affects the Back orientation severely
%--------------------------------------------------------------------------


phase_use = flipdim(NFM_REG,2);
mask_use = flipdim(msk,2);

% multicos6 = rot90(multicos6, 1);

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

imagesc3d2(chi_cosmos - ~chi_cosmos, N/2, 12, [90,90,-90], [-0.1,0.1])

% imagesc3d2(magn_use, N/2, 13, [90,90,-90], [0,0.5])
% 

 
