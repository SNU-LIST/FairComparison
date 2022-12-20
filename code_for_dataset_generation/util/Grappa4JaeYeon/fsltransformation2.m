function[] = fsltransformation2(fname,mat,rotcenter);
%function[] = fsltransformatin(fname,mat,rotcenter);
%mat = [xshift yshift zshift xrot yrot zrot]; shift is in mm rotation is in angle

% xshift = mat(1);
% yshift = mat(2);
% zshift = mat(3);
% xrot   = mat(4)*pi/180;
% yrot   = mat(5)*pi/180;
% zrot   = mat(6)*pi/180;

% shiftmat = [1 0 0 xshift;0 1 0 yshift;0 0 1 zshift;0 0 0 1];
% xrotmat = [1 0 0 0;0 cos(xrot) -sin(xrot) 0;0 sin(xrot) cos(xrot) 0;0 0 0 1];
% yrotmat = [cos(yrot) 0 sin(yrot) 0;0 1 0 0;-sin(yrot) 0 cos(yrot) 0;0 0 0 1];
% zrotmat = [cos(zrot) -sin(zrot) 0 0;sin(zrot) cos(zrot) 0 0;0 0 1 0;0 0 0 1];

% if nargin <3
% rotmat = mat;
% else
% shiftmat2 = [1 0 0 -rotcenter(1);0 1 0 -rotcenter(2);0 0 1 -rotcenter(3);0 0 0 1]; 
% rotmat = inv(shiftmat2)*mat*shiftmat2;
% end
rotmat = mat;
fid = fopen(fname,'wt');
fprintf(fid,'%3.6f %3.6f %3.6f %3.6f\n ',rotmat(1,:));
fprintf(fid,'%3.6f %3.6f %3.6f %3.6f\n ',rotmat(2,:));
fprintf(fid,'%3.6f %3.6f %3.6f %3.6f\n ',rotmat(3,:));
fprintf(fid,'%3.6f %3.6f %3.6f %3.6f\n ',rotmat(4,:));
fclose(fid);