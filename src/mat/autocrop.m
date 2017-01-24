% Author: Qinghui Liu @ USN Norway
% 2017-01-17, version 1.0
% Can wrap into a function in future
% automatically crop .tiff images to 320*320 
% save as png formate for caffe input dataset

clc
clear
file_adi='/Users/liuqh/Documents/MATLAB/trainset'; % change to your dataset folder
cd(file_adi);
photofle = dir('*.tiff'); % change 

[file_num,dn]=size(photofle)

Xstart = 1   % start X coord for crop func
Ystart = 1   % start Y coord for crop func
width  = 319 % rect size for crop func
height = 319 % rect size for crop func

for cur=1:file_num

    [X, map] = imread(photofle(cur,1).name);
    %[Hrow Wcol Dim] = size(X)

    Y = imcrop(X,[Xstart Ystart width height]); 

    imwrite(Y,[photofle(cur,1).name '.png'])

end