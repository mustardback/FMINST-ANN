%Simple back propagation test to classify images of the numbers 0, 1, and 2
function [output] = testBackProp()

%Create prototypes for numbers 0, 1, and 2 for training
zeroPrototype = [0;  1;  1;  1;  1; 0;
                  1; 0; 0; 0; 0;  1;
                  1; 0; 0; 0; 0;  1;
                  1; 0; 0; 0; 0;  1;
                 0;  1;  1;  1;  1; 0;];
             
onePrototype = [0; 0; 0; 0; 0; 0;  
                 1; 0; 0; 0; 0; 0; 
                 1;  1;  1;  1;  1;  1; 
                0; 0; 0; 0; 0; 0;
                0; 0; 0; 0; 0; 0;];
            
twoPrototype = [ 1; 0; 0; 0; 0; 0;
                 1; 0; 0;  1;  1;  1;
                 1; 0; 0;  1; 0;  1;
                 0; 1;  1; 0; 0; 1;
                 0; 0; 0; 0; 0; 1];

%create labels for training network             
zeroLabel = [1;0;0];
oneLabel  = [0;1;0];
twoLabel = [0;0;1];

%Add prototypes and labels to training matrices
trainingMatrix = [zeroPrototype onePrototype twoPrototype];
trainingLabels = [zeroLabel, oneLabel, twoLabel];

%test numbers with noise

%create noisy images
badPrototypeZero = zeroPrototype;   
for i = 1 : 8
    rand = randi([1 30],1,1);
    if badPrototypeZero(rand) == 0
       badPrototypeZero(rand) = 1;
    else
       badPrototypeZero(rand) = 0;
    end
end

badPrototypeOne = onePrototype;   
for i = 1 : 8
    rand = randi([1 30],1,1);
    if badPrototypeOne(rand) == 0
       badPrototypeOne(rand) = 1;
    else
       badPrototypeOne(rand) = 0;
    end
end

badPrototypeTwo = twoPrototype;   
for i = 1 : 8
    rand = randi([1 30],1,1);
    if badPrototypeTwo(rand) == 0
       badPrototypeTwo(rand) = 1;
    else
       badPrototypeTwo(rand) = 0;
    end
end

%Create matrix for noisy images and their correct output
fourBadPixels = [badPrototypeZero badPrototypeOne badPrototypeTwo];
fourBadLabels = trainingLabels;

%Run back propagation on data
%Backprop arguments
%(trainingImages, trainingLabels, numEpochs, numOutputs, trainingPerEpoch, testImages, testLabels)
Backprop(trainingMatrix, trainingLabels, 50, 3, 5, fourBadPixels, fourBadLabels);
