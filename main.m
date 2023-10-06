% data: 2023-10-04
% version: 1.0
% programmer: xlxlqqq
% function: train a CNN to detect handwriting number

clear;
clc;
close all;

filename = 'mnist0.mat';
load(filename);

traindatatable = table(trainimages, trainlabels);
testdatatable = table(testimages, testlabels);

%% designed by self
layers = [
    imageInputLayer([28 28 1])         % 选择zerocenter归一化方法
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10);
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', testdatatable, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% %% designed accroding to Lenet  
% layers= [
%    imageInputLayer([28 28 1],'Name','input','Normalization','zerocenter')                  
%    convolution2dLayer([5 5],6,'Padding','same','Name','Conv1') 
%    maxPooling2dLayer(2,'Stride',2,'Name','Pool1')              
%    convolution2dLayer([5 5],16,'Padding','same','Name','Conv2')
%    maxPooling2dLayer(2,'Stride',2,'Name','Pool2')              
%    convolution2dLayer([5 5],120,'Padding','same','Name','Conv3')
%    fullyConnectedLayer(10,'Name','fc1')
%    softmaxLayer( 'Name','softmax')
%    classificationLayer('Name','output') 
%  ]; 
%                                      
%  options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.0005, ...    
%     'MaxEpochs',2, ... 
%     'Shuffle','every-epoch', ...
%     'ValidationData',testdatatable, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');  


% %% KNN的识别
% matrix = [];% 训练矩阵
% for delta = 0:9%构建训练区样本的矩阵
%   label_path = strcat('D:\engeering lib\matlab2019a\matlabfuns\CNN\mnist_train\',int2str(delta),'\');
%   disp(length(dir([label_path '*.png'])));
%   for i = 1:length(dir([label_path '*.png']))
%         im = imread(strcat(label_path,'\',int2str(delta),'_',int2str(i-1),'.png'));
%         %imshow(im);
%         im = imbinarize(im);%图像二值化
%         temp = [];
%         for j = 1:size(im,1)% 训练图像行向量化
%             temp = [temp,im(j,:)];
%         end
%         matrix = [matrix;temp];
%   end
% end
% 
% label = [];%在标签矩阵后添加标签列向量
%  for i = 0:9
%     tem = ones(length(dir([label_path '*.png'])),1) * i;
%     label = [label;tem];
% end
% matrix = horzcat(matrix,label);%带标签列的训练矩阵
% 
% %测试对象向量
% for delta = 0:9%构建测试图像的向量
%     test_path = strcat('C:\Users\ABC\Desktop\KNN\test\',int2str(delta),'\');
%     len = (length(dir([test_path '*.png'])));
%     disp(len);
%     p = 0;% 识别结果计数
%     for i = 1:len
%         vec = []; %　测试样本行向量化       
%         test_im = imread(strcat('test2\',int2str(delta),'\',int2str(delta),'_',int2str(i-1),'.png'));
%         imshow(test_im);
%         test_im = imbinarize(test_im);
%         for j = 1:size(test_im,1)
%             vec = [vec,test_im(j,:)];
%         end
% 
%         dis = [];
%         for count = 1:length(dir([label_path '*.png'])) * 10
%             row = matrix(count,1:end-1);% 不带标签的训练矩阵每一行向量
%             distance = norm(row(1,:)-vec(1,:));% 求欧氏几何距离
%             dis = [dis;distance(1,1)];% 距离列向量
%         end
%         test_matrix = horzcat(matrix,dis);% 加入表示距离的列向量
% 
% 
%         %排序
%         test_matrix = sortrows(test_matrix,size(test_matrix,2));
%         %输入K值，前K个行向量标签的众数作为结果输出
%         K = 5;
%         result = mode(test_matrix(1:K,end-1));
%         disp(strcat('图像',int2str(delta),'_',int2str(i),'.png','的识别结果是：',int2str(result)));
% 
%         if(delta == result)
%             p = p + 1;
%         end
%     end
%     pi = p/len;
%     disp(strcat('识别精度为：',num2str(pi)));
%     disp('Finished!'); 
% end


%% 
net = trainNetwork(traindatatable, layers, options);

analyzeNetwork(net);

labelsp = classify(net, testdatatable);

temp = find(labelsp ~= testlabels);
for index = 1:4
   subplot(2, 2, index)
   imshow(testimages{temp(index)});
   title(labelsp(temp(index)))
end

accuracy = sum(labelsp == testlabels) / numel(testlabels);






