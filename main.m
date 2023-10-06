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
    imageInputLayer([28 28 1])         % ѡ��zerocenter��һ������
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


% %% KNN��ʶ��
% matrix = [];% ѵ������
% for delta = 0:9%����ѵ���������ľ���
%   label_path = strcat('D:\engeering lib\matlab2019a\matlabfuns\CNN\mnist_train\',int2str(delta),'\');
%   disp(length(dir([label_path '*.png'])));
%   for i = 1:length(dir([label_path '*.png']))
%         im = imread(strcat(label_path,'\',int2str(delta),'_',int2str(i-1),'.png'));
%         %imshow(im);
%         im = imbinarize(im);%ͼ���ֵ��
%         temp = [];
%         for j = 1:size(im,1)% ѵ��ͼ����������
%             temp = [temp,im(j,:)];
%         end
%         matrix = [matrix;temp];
%   end
% end
% 
% label = [];%�ڱ�ǩ�������ӱ�ǩ������
%  for i = 0:9
%     tem = ones(length(dir([label_path '*.png'])),1) * i;
%     label = [label;tem];
% end
% matrix = horzcat(matrix,label);%����ǩ�е�ѵ������
% 
% %���Զ�������
% for delta = 0:9%��������ͼ�������
%     test_path = strcat('C:\Users\ABC\Desktop\KNN\test\',int2str(delta),'\');
%     len = (length(dir([test_path '*.png'])));
%     disp(len);
%     p = 0;% ʶ��������
%     for i = 1:len
%         vec = []; %������������������       
%         test_im = imread(strcat('test2\',int2str(delta),'\',int2str(delta),'_',int2str(i-1),'.png'));
%         imshow(test_im);
%         test_im = imbinarize(test_im);
%         for j = 1:size(test_im,1)
%             vec = [vec,test_im(j,:)];
%         end
% 
%         dis = [];
%         for count = 1:length(dir([label_path '*.png'])) * 10
%             row = matrix(count,1:end-1);% ������ǩ��ѵ������ÿһ������
%             distance = norm(row(1,:)-vec(1,:));% ��ŷ�ϼ��ξ���
%             dis = [dis;distance(1,1)];% ����������
%         end
%         test_matrix = horzcat(matrix,dis);% �����ʾ�����������
% 
% 
%         %����
%         test_matrix = sortrows(test_matrix,size(test_matrix,2));
%         %����Kֵ��ǰK����������ǩ��������Ϊ������
%         K = 5;
%         result = mode(test_matrix(1:K,end-1));
%         disp(strcat('ͼ��',int2str(delta),'_',int2str(i),'.png','��ʶ�����ǣ�',int2str(result)));
% 
%         if(delta == result)
%             p = p + 1;
%         end
%     end
%     pi = p/len;
%     disp(strcat('ʶ�𾫶�Ϊ��',num2str(pi)));
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






