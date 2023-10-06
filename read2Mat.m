% data: 2023-10-04
% version: 1.0
% xlxlqqq
% function: read idx files and save to mat files

clear;
clc;
close all;

%% read trainimages
filename1 = 'train-images.idx3-ubyte';

fid = fopen(filename1);
magic1 = fread(fid, 4);
num1 = fread(fid, 4);
row1 = fread(fid, 4);
colomn1 = fread(fid, 4);
magic1 = transfer(magic1);
num1 = transfer(num1);
row1 = transfer(row1);
colomn1 = transfer(colomn1);


trainimages = cell(num1, 1);
for i = 1:num1
    temp = fread(fid, row1 * colomn1);
    temp = reshape(temp, [row1, colomn1]);
    trainimages{i} = temp';
end

fclose(fid);

index = 5;
figure
imshow(trainimages{index});

%% read trainlabels
filename2 = 'train-labels.idx1-ubyte';

fid = fopen(filename2);
magic2 = fread(fid, 4);
num2 = fread(fid, 4);
magic2 = transfer(magic2);
num2 = transfer(num2);

trainlabels = zeros(num2, 1);

for i = 1 : num2
   trainlabels(i) = fread(fid, 1);
end
trainlabels = categorical(trainlabels);
fclose(fid);


%% read test images
filename1 = 't10k-images.idx3-ubyte';

fid = fopen(filename1);
magic1 = fread(fid, 4);
num1 = fread(fid, 4);
row1 = fread(fid, 4);
colomn1 = fread(fid, 4);
magic1 = transfer(magic1);
num1 = transfer(num1);
row1 = transfer(row1);
colomn1 = transfer(colomn1);


testimages = cell(num1, 1);
for i = 1:num1
    temp = fread(fid, row1 * colomn1);
    temp = reshape(temp, [row1, colomn1]);
    testimages{i} = temp';
end
fclose(fid);

index = 5;
figure
imshow(testimages{index});

%% read test labels
filename2 = 't10k-labels.idx1-ubyte';

fid = fopen(filename2);
magic2 = fread(fid, 4);
num2 = fread(fid, 4);
magic2 = transfer(magic2);
num2 = transfer(num2);

testlabels = zeros(num2, 1);

for i = 1 : num2
   testlabels(i) = fread(fid, 1);
end
testlabels = categorical(testlabels);
fclose(fid);


%% save files data as mat files
save mnist0.mat trainimages trainlabels testimages testlabels
% a = load('mnist0.mat');



%% function transfer bin to dec
function y = transfer( data )
    b = dec2bin(data, 8);
    c = [b(1, :), b(2, :), b(3, :), b(4, :)];
    y = bin2dec(c);
end



