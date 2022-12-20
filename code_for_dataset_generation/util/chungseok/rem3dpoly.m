function[img_out] = rem3dpoly(img_in,mask,rem_DC_offset)
if nargin < 3
	rem_DC_offset = 1;
end

[sx,sy,sz] = size(img_in);
idx = find(mask~=0);
si = length(idx);
img_ini = img_in(idx);

% 0th order
if rem_DC_offset == 1
f0 = ones(sx,sy,sz);
else
f0 = zeros(sx,sy,sz); 
end
% 1st order
[fx,fy,fz] = meshgrid(linspace(-1,1,sy),linspace(-1,1,sx),linspace(-1,1,sz));
% 2nd order
fx2 = fx.^2;
fy2 = fy.^2;
fz2 = fz.^2;
fxy = fx.*fy;
fzx = fz.*fx;
fzy = fz.*fy;
% 3rd order
fx3 = fx.^3;
fy3 = fy.^3;
fz3 = fz.^3;
fx2y = fx.^2.*fy;
fy2x = fy.^2.*fx;
fz2x = fz.^2.*fx;
fx2z = fx.^2.*fz;
fz2y = fz.^2.*fy;
fy2z = fy.^2.*fz;
fxyz = fx.*fy.*fz;
% 4th order
fx4 = fx.^4;
fy4 = fy.^4;
fz4 = fz.^4;
fx3y = fx.^3.*fy;
fy3x = fy.^3.*fx;
fz3x = fz.^3.*fx;
fx3z = fx.^3.*fz;
fz3y = fz.^3.*fy;
fy3z = fy.^3.*fz;
fx2y2 = fx.^2.*fy.^2;
fz2x2 = fz.^2.*fx.^2;
fz2y2 = fz.^2.*fy.^2;
fx2yz = fx.^2.*fy.*fz;
fy2zx = fy.^2.*fz.*fx;
fz2xy = fz.^2.*fx.*fy;

f0i = f0(idx);
fxi =fx(idx); fyi=fy(idx); fzi=fz(idx);
fx2i = fx2(idx); fy2i = fy2(idx); fz2i = fz2(idx);
fxyi=fxy(idx); fzxi=fzx(idx); fzyi=fzy(idx);
fx3i = fx3(idx); fy3i = fy3(idx); fz3i=fz3(idx);
fx2yi=fx2y(idx); fz2xi=fz2x(idx); fz2yi=fz2y(idx);
fy2xi=fy2x(idx); fx2zi=fx2z(idx); fy2zi=fy2z(idx);
fxyzi = fxyz(idx);
fx4i = fx4(idx); fy4i = fy4(idx); fz4i=fz4(idx);
fx3yi=fx3y(idx); fz3xi=fz3x(idx); fz3yi=fz3y(idx);
fy3xi=fy3x(idx); fx3zi=fx3z(idx); fy3zi=fy3z(idx);
fx2y2i=fx2y2(idx); fz2x2i=fz2x2(idx); fz2y2i=fz2y2(idx);
fy2zxi=fy2zx(idx); fx2yzi=fx2yz(idx); fz2xyi=fz2xy(idx);


Atmp = [f0i, fxi, fyi, fzi, fx2i, fy2i, fz2i, fxyi, fzxi, fzyi, ... 
	fx3i, fy3i, fz3i, fx2yi, fy2xi, fz2xi, fx2zi, fz2yi, fy2zi, fxyzi, ...
    fx4i, fy4i, fz4i, fx3yi, fy3xi, fz3xi, fx3zi, fz3yi, fy3zi, ...,
    fx2y2i, fz2x2i, fz2y2i, fy2zxi, fx2yzi, fz2xyi, ...,
    ];

x = -pinv(Atmp)*img_ini(:);

img_out = (f0*x(1) + fx*x(2) + fy*x(3) + fz*x(4) + ...
+ fx2*x(5) + fy2*x(6) + fz2*x(7) + fxy*x(8) + fzx*x(9) + fzy*x(10) + ...
+ fx3*x(11) + fy3*x(12) + fz3*x(13) + fx2y*x(14) + fy2x*x(15) + fz2x*x(16) + ...
+ fx2z*x(17) + fz2y*x(18) + fy2z*x(19) + fxyz*x(20) + ...
+ fx4*x(21) + fy4*x(22) + fz4*x(23) + fx3y*x(24) + fy3x*x(25) + fz3x*x(26) + ...
+ fx3z*x(27) + fz3y*x(28) + fy3z*x(29) + fx2y2*x(30) + fz2x2*x(31) + fz2y2*x(32) + ...
+ fy2zx*x(33) + fx2yz*x(34) + fz2xy*x(35) + ...
+ img_in).*mask;

