function res = ifft2c(x)
fctr = size(x,1)*size(x,2);
for n=1:size(x,3)
res(:,:,n) = ifftshift(fft2(fftshift(x(:,:,n))))/sqrt(fctr);
end
