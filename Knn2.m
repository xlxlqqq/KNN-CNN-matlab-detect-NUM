% data:2023-10-06
% programmer: xlxlqqq
% function: detect Arabic numerals with the way of KNN(binarization)

clear;
clc;
close all;

load('mnist0.mat');

trainDataMatrix = zeros(length(trainimages), 28 * 28);
testDataMatrix = zeros(length(testimages), 28 * 28);

for index = 1:length(trainimages)
    for j = 1:28
       trainDataMatrix(index, ((j-1) * 28 + 1) : (28 * j)) = trainimages{index}(j, :);
    end
end

for index = 1:length(testimages)
    for j = 1:28
       testDataMatrix(index, ((j-1) * 28 + 1) : (28 * j)) = testimages{index}(j, :);
    end
end


for i = 1 : length(trainimages)
   for j = 1 : 28 * 28
       if trainDataMatrix(i, j) >= 50
           trainDataMatrix(i, j) = (1);
       else
           trainDataMatrix(i, j) = (0);
       end
   end
end

for i = 1 : length(testimages)
   for j = 1 : 28 * 28
       if testDataMatrix(i, j) >= 50
           testDataMatrix(i, j) = (1);
       else
           testDataMatrix(i, j) = (0);
       end
   end
end

trainDataMatrix = logical(trainDataMatrix);
testDataMatrix = logical(testDataMatrix);



a = testDataMatrix(1, :);
b = trainDataMatrix(1, :);
z = a - b;
D = sum(z .* z);


decNum = 0;
errorNum = 0;
errorList = zeros(1000, 1);
testNum = length(testimages) / 100;
distance = zeros(length(trainimages), 1);
for index = 1 : testNum
    for j = 1 : (length(trainimages) / 1)
        z = testDataMatrix(index, :) - trainDataMatrix(j, :);
        distance(j, 1) = sum(z .* z);
        index
        j
    end
    
    [min, minkindex] = mink(distance, 5);
    output = mode(trainlabels(minkindex));

    if output == testlabels(index)
        decNum = decNum + 1; 
    else
        errorNum = errorNum + 1;
        errorList(errorNum) = index;
    end
end

accuracy = decNum / (testNum);
disp(accuracy);

% imshow(testimages{errorList(5)});
