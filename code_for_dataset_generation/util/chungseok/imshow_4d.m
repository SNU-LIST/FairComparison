function  [] = imshow_4d( matrix, disprange )
N = size(matrix);
im = permute(reshape(permute(matrix,[1 3 2 4]),[N(1),N(3),N(2)*N(4)]),[1 3 2]);
if (nargin < 2)
    imshow_3d(im, [])
else
    imshow_3d(im, disprange)
end
end