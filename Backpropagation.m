% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
trainingImages = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

trainingImages = trainingImages.';
trainingLabels = trainingLabels.';
testImages = testImages.';
testLabel = testLabel.';


%TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
learningRate = 0.1;
epochs = 50;

layerOneNeurons = 24;
layerTwoNeurons = 24;
outputNeurons = 1;
inputLength = 784;
layerOneMatrix = randn(inputLength,layerOneNeurons);
wH = randn(inputLength,layerOneNeurons);
biasLayerOne = randn(1,layerOneNeurons);

layerTwoMatrix = randn(layerOneNeurons,layerTwoNeurons);
wH2 = randn(layerOneNeurons,layerTwoNeurons);
biasLayerTwo = randn(1,layerTwoNeurons);

outputMatrix = randn(layerTwoNeurons,outputNeurons);
wO = randn(layerTwoNeurons,outputNeurons);
biasOutputLayer = randn(1,outputNeurons);

layerOneMatrix = layerOneMatrix./sqrt(layerOneNeurons);
layerTwoMatrix = layerTwoMatrix./sqrt(layerTwoNeurons);
outputMatrix = outputMatrix./sqrt(outputNeurons);

E = zeros(1,epochs);
tic;
disp('Training Network');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:length(trainingImages)
           
            % Random dataset row selection parameter
            r = round(length(trainingImages).*rand(1,1));
            if r == 0
                r = r+1;
            end

            % FORWARD PASS
            % first layer layerOneOutput
            layerOneOutput = trainingImages(r,:)*layerOneMatrix; %WP
            layerOneOutput = layerOneOutput + biasLayerOne;      %+b = n
            layerOneSig = logsig(layerOneOutput);                %a^1 = logsim(n)
%second layer layerTwoOutput
            layerTwoOutput = layerOneSig*layerTwoMatrix;         %Wp
            layerTwoOutput = layerTwoOutput + biasLayerTwo;      %+b = n
            layerTwoSig = logsig(layerTwoOutput);                %a^2 = logsig(n)
            
            %output layer layerThreeOutput layerThreeLinear
            layerThreeOutput = layerTwoSig*outputMatrix;        %Wp
            layerThreeOutput = layerThreeOutput+biasOutputLayer;%+b = n
            layerThreeLinear = layerThreeOutput;                %a^3 = linear(n)
            
            % ERROR CALCULATION (target - output) 
            e = trainingLabels(:,r)-layerThreeLinear;
            
            % BACKWARD PASS
            
            %output sensitivity = -2(1) * error
            outputSensitivity = -2*e;
            biasOutputLayer = biasOutputLayer - (learningRate*outputSensitivity);
            
            %second layer sensitivity = layerTwoS
            layerTwoS = diag(dlogsig(layerTwoOutput,layerTwoSig))*(outputMatrix*outputSensitivity');
            
            %first layer sensitivity = layerOneS
            layerOneS  = diag(dlogsig(layerOneOutput,layerOneSig))*(layerTwoMatrix*layerTwoS);           
            
            outputMatrix  = outputMatrix -(learningRate.*outputSensitivity'*layerTwoSig)';

            layerTwoMatrix = layerTwoMatrix -(learningRate.*layerTwoS*layerOneSig)';          
            biasLayerTwo = biasLayerTwo' + (learningRate*layerTwoS);
            biasLayerTwo = biasLayerTwo';
            
            layerOneMatrix  = layerOneMatrix -(learningRate.*layerOneS*trainingImages(r,:)).';
            biasLayerOne = biasLayerOne' + (learningRate*layerOneS);
            biasLayerOne = biasLayerOne';
            E(t) = E(t) + (e.^2);
    end
E(t) = E(t)/length(trainingImages);

end
toc;

figure;
plot(1:t,E,'b--o')
title('Error Curve');
xlabel('Epochs');
ylabel('Squared Error Sum');

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
