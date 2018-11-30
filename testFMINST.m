function [output] = testFMINST()

trainFMINSTMatrix = csvread('train.csv',1,1);

validationFMINSTImages = loadValidationImages(trainFMINSTMatrix);
validationFMINSTLabels = loadValidationLabels(trainFMINSTMatrix);
trainingFMINSTImages = loadTrainingImages(trainFMINSTMatrix);
trainingFMINSTLabels = loadTrainingLabels(trainFMINSTMatrix); 


%create labels for training network             
%Create training matrix for labels that is compatible with 10 output network
trainingLabelVectors = zeros(10, length(trainingFMINSTLabels));
for t = 1:length(trainingFMINSTLabels)
    target = trainingFMINSTLabels(t);
    for i = 1:10
       if i == (target+1)
           trainingLabelVectors(i,t) = 1;
       end
       if i ~= (target+1)
          trainingLabelVectors(i,t) = 0; 
       end
    end
end

%Create test matrix for labels that is compatible with 10 output network
testLabelVectors = zeros(10, length(validationFMINSTLabels));
for t = 1:length(validationFMINSTLabels)
    target = validationFMINSTLabels(t);
    for i = 1:10
       if i == (target+1)
           testLabelVectors(i,t) = 1;
       end
       if i ~= (target+1)
          testLabelVectors(i,t) = 0; 
       end
    end
end

function images = loadTrainingImages(filename)
    %loadFMNISTImages returns a 28x28x[number of MNIST images]x0.9 matrix containing
    %the raw FMNIST images
    images = filename(1:50000, 2:end);
    images = permute(images,[2 1 3]);
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;
end

function labels = loadTrainingLabels(filename)
%loadFMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the FMNIST images
    labels = filename(1:50000, 1);
end

function images = loadValidationImages(filename)
    %loadFMNISTImages returns a 28x28x[number of FMNIST images] matrix containing
    %the raw FMNIST images
    images = filename(50001:end,2:end);
    images = permute(images,[2 1 3]);
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;
end

    function labels = loadValidationLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images
    labels = filename(50001:end, 1);
end

%Run back propagation on data
%Backprop arguments
%(trainingImages, trainingLabels, numEpochs, numOutputs, trainingPerEpoch, testImages, testLabels)
Backprop(trainingFMINSTImages, trainingLabelVectors, 20, 10, 25, validationFMINSTImages, testLabelVectors);
end