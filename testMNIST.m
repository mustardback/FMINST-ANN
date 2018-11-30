function [testMNIST] = testMNIST()

%Import MNIST training and test data

trainingImages = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%Create training matrix for labels that is compatible with 10 output network
trainingLabelVectors = zeros(10, length(trainingLabels));

for t = 1:length(trainingLabels)
    target = trainingLabels(t);
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
testLabelVectors = zeros(10, length(testLabels));
for t = 1:length(testLabels)
    target = testLabels(t);
    for i = 1:10
       if i == (target+1)
           testLabelVectors(i,t) = 1;
       end
       if i ~= (target+1)
          testLabelVectors(i,t) = 0; 
       end
    end
end
function images = loadMNISTImages(filename)
    %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    %the raw MNIST images
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);
    fclose(fp);
    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;
end
function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);
end

%Run back propagation on data
%Backprop arguments
%(trainingImages, trainingLabels, numEpochs, numOutputs, trainingPerEpoch, testImages, testLabels)

Backprop(trainingImages, trainingLabelVectors, 100, 10, length(trainingLabels), testImages, testLabelVectors);

end