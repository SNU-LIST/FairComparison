clear all
addpath(genpath('../../Code/util'))

% eul=[rand(1)*pi*2, rand(1)*pi/6, rand(1)*pi*2]
eul=[
    4.3305    0.2879    3.6141; % train1 - 1dir
    4.7008    0.0759    0.3756; % train1 - 2dir
    2.8308    0.4466    1.4752; % ...
    0.5267    0.3257    2.2190;
    1.4387    0.1838    5.1597;
    5.7387    0.2687    0.0968;
    0.9574    0.2104    0.2703;
    5.1888    0.0398    1.0618;
    3.3825    0.1256    4.0785;
    6.2589    0.0646    4.5975;
    0.4912    0.0963    4.0699;
    2.7814    0.1256    2.8332;
    0.6701    0.2185    3.4370;
    6.0438    0.0260    1.8618;
    0.0291    0.4727    4.6790;
    4.8689    0.4947    1.1872;
    5.1353    0.2570    4.3151;
    5.4582    0.2562    1.1530;
    0.5305    0.1768    2.3153;
    2.5119    0.4713    3.9309;
    1.6328    0.1933    4.9023;
    5.0270    0.0582    0.5097;
    2.7107    0.4085    5.8395;
    5.7218    0.2041    4.8739;
    1.1426    0.1265    3.0586;
    1.6575    0.2115    2.7386;
    0.9144    0.0505    2.8072;
    0.8549    0.0691    1.9249;
    5.4619    0.4933    3.1951;
    3.6424    0.5006    3.2093;];

for i1=2:6
for i2=1:5
rotm= eul2rotm(eul((i1-1)*5+i2,:), 'ZYZ');
M = [1 0 0 88; 0 1 0 88; 0 0 1 27; 0 0 0 1]*[rotm, zeros(3, 1);zeros(1, 3) 1]*[1 0 0 -88; 0 1 0 -88; 0 0 1 -27; 0 0 0 1];
v = M(1:3,1:3) * [0;0;1];
disp(['Rotation: ', num2str(acosd(v(3))), ' degrees'])
fsltransformation2(['train' num2str(i1) '/regisyn' num2str(i2)], M)
end
end
spatial_res = [1 1 3];
mkdir('result')


%% 
for N=2:6
    cd(['train' num2str(N)])

    load('phscos_crop.mat','multicos','multimask');
    save_nii(make_nii(single(fliplr(multicos(:,:,:,1))), spatial_res, [], 16), 'cosmos.nii');
    save_nii(make_nii(single(fliplr(multimask(:,:,:,1))), spatial_res, [], 16), 'mask.nii');

    cmd = {};
    for t=1:5
    n=num2str(t);
    cmd = [cmd, ['flirt -in cosmos  -ref cosmos -applyxfm -init regisyn' n ' -interp sinc -sincwidth 15 -sincwindow hanning -out cosmossyn' n]];
    cmd = [cmd, ['flirt -in mask    -ref mask   -applyxfm -init regisyn' n ' -interp sinc -sincwidth 15 -sincwindow hanning -out masksyn' n]];
    end
    system_pl(cmd{1},cmd{2},cmd{3},cmd{4},cmd{5},cmd{6},cmd{7},cmd{8},cmd{9},cmd{10});
    system('gunzip -f *.gz');

    for t=1:5
    data = load_nii(['cosmossyn' num2str(t) '.nii']);
    multicossyn(:,:,:,t) = fliplr(data.img);
    data = load_nii(['masksyn' num2str(t) '.nii']);
    multimasksyn(:,:,:,t) = fliplr(data.img);
    end

    save phscos_aug.mat multicossyn multimasksyn     
    cd ..
end

%% 
for N=2:6
    load(['train' num2str(N) '/phscos_crop.mat'])
    load(['train' num2str(N) '/phscos_aug.mat'])
    d = dipole_kernel([176,176,54], spatial_res, 3, 'kspace');
    for ii=[6,7,8,9,10]
        multicos(:,:,:,ii) = multicossyn(:,:,:,ii-5);
        multimask(:,:,:,ii) = multimasksyn(:,:,:,ii-5)>0.7;
        multiphs(:,:,:,ii) = ifftn(fftn(multicos(:,:,:,ii)).*d).*multimask(:,:,:,ii);
    end
    %gcf=figure; imshow_4d(cat(1,multicos,multiphs*3,multimask),[-0.15,0.15])
    %saveas(gcf,['result/train' num2str(N) '.jpg'])
    save(['result/train' num2str(N) '.mat'], 'multicos', 'multimask', 'multiphs')  
end


%% 
for N=7
    load(['train' num2str(N) '/phscos_crop.mat'])
    %gcf=figure; imshow_4d(cat(1,multicos,multiphs*3,multimask),[-0.15,0.15])
    %saveas(gcf,['result/train' num2str(N) '.jpg'])
    save(['result/train' num2str(N) '.mat'], 'multicos', 'multimask', 'multiphs')    
end

for N=1:5
    load(['test' num2str(N) '/phscos_crop.mat'])
    %gcf=figure; imshow_4d(cat(1,multicos,multiphs*3,multimask),[-0.15,0.15])
    %saveas(gcf,['result/test' num2str(N) '.jpg'])
    save(['result/test' num2str(N) '.mat'], 'multicos', 'multimask', 'multiphs')    
end

