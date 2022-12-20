unix('gunzip *.gz')

mag_notrot_norm2 = zeros(size(mag_notrot));

data = load_nii('Abslow_N_bet_bias.nii');
bias = data.img;
f = fittype('a*x+b');
msk = MSK_low(:, :, :, 1);
mag_temp = mag_notrot(:, :, :, 1);
[result,good]=fit(bias(msk>0),mag_temp(msk>0),f);
bias(:)=result.a.*bias(:)+result.b;
mag_temp(msk>0) = mag_temp(msk>0)./bias(msk>0);
mag_notrot_norm2(:, :, :, 1) = mag_temp;

data = load_nii('Abslow_L_bet_bias.nii');
bias = data.img;
f = fittype('a*x+b');
msk = MSK_low(:, :, :, 2);
mag_temp = mag_notrot(:, :, :, 2);
[result,good]=fit(bias(msk>0),mag_temp(msk>0),f);
bias(:)=result.a.*bias(:)+result.b;
mag_temp(msk>0) = mag_temp(msk>0)./bias(msk>0);
mag_notrot_norm2(:, :, :, 2) = mag_temp;

data = load_nii('Abslow_R_bet_bias.nii');
bias = data.img;
f = fittype('a*x+b');
msk = MSK_low(:, :, :, 3);
mag_temp = mag_notrot(:, :, :, 3);
[result,good]=fit(bias(msk>0),mag_temp(msk>0),f);
bias(:)=result.a.*bias(:)+result.b;
mag_temp(msk>0) = mag_temp(msk>0)./bias(msk>0);
mag_notrot_norm2(:, :, :, 3) = mag_temp;

data = load_nii('Abslow_B_bet_bias.nii');
bias = data.img;
f = fittype('a*x+b');
msk = MSK_low(:, :, :, 4);
mag_temp = mag_notrot(:, :, :, 4);
[result,good]=fit(bias(msk>0),mag_temp(msk>0),f);
bias(:)=result.a.*bias(:)+result.b;
mag_temp(msk>0) = mag_temp(msk>0)./bias(msk>0);
mag_notrot_norm2(:, :, :, 4) = mag_temp;

data = load_nii('Abslow_F_bet_bias.nii');
bias = data.img;
f = fittype('a*x+b');
msk = MSK_low(:, :, :, 5);
mag_temp = mag_notrot(:, :, :, 5);
[result,good]=fit(bias(msk>0),mag_temp(msk>0),f);
bias(:)=result.a.*bias(:)+result.b;
mag_temp(msk>0) = mag_temp(msk>0)./bias(msk>0);
mag_notrot_norm2(:, :, :, 5) = mag_temp;