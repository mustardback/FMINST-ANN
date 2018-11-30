%Backprop takes a training set and trains a Backprop Neural Network
%Description of arguments:
%trainingImages: A matrix of images used to train network
%trainingLabels: The corresponding correct output for training images
%epoch: The number of times to run through training data to train network
%numOutputs: The number of output neurons, must match row axis of
%trainingLabels
%trainingPerEpoch: the number of times to train network per epoch
%testImages: Images to test trained network on
%testLabels: The corresponding correct outputs for test images
function [output] = Backprop(trainingImages, trainingLabels, epochs, numOutputs, trainingPerEpoch, testImages, testLabels)
learningRate = 0.001;

layerOneNeurons = 100;
layerTwoNeurons = 100;
outputNeurons = numOutputs;

%get the number of elements in each column vector for input
inputLength = size(trainingImages,1);

%Create weight matrix and biases for first layer of network
%Generate random values for matrices
layerOneMatrix = randn(layerOneNeurons,inputLength);
biasLayerOne = randn(1,layerOneNeurons);

%Create weight matrix and biases for second layer of network
%Generate random values for matrices
layerTwoMatrix = randn(layerTwoNeurons,layerOneNeurons);
biasLayerTwo = randn(1,layerTwoNeurons);

%Create weight matrix and biases for output layer
%Generate random values for matrices
outputMatrix = randn(outputNeurons,layerTwoNeurons);
biasOutputLayer = randn(1,outputNeurons);

%Square root each value for weight matrices to start with small weights
layerOneMatrix = layerOneMatrix./sqrt(layerOneNeurons);
layerTwoMatrix = layerTwoMatrix./sqrt(layerTwoNeurons);
outputMatrix = outputMatrix./sqrt(outputNeurons);

%Generate error vector
e = zeros(1,numOutputs);

%Create vector to store sum of error squared for training and test sets
E = zeros(1,epochs);
testE = zeros(1,epochs);

%Start timer to see how long training takes
tic;
disp('Training Network');

%Train network epoch number of times
for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    numElements = size(trainingImages,2); %Get number of elements in training set
    for k = 1:trainingPerEpoch
            
            % Select random image from training set
            r = round(numElements.*rand(1,1));
            if r == 0
                r = r+1;
            end

            % FORWARD PASS
            % first layer layerOneOutput
            layerOneOutput = layerOneMatrix * trainingImages(:,r); %WP
            layerOneOutput = layerOneOutput + biasLayerOne';       %+b = n
            layerOneSig = logsig(layerOneOutput);                  %a^1 = logsim(n)   
            
            %second layer layerTwoOutput
            layerTwoOutput =layerTwoMatrix * layerOneSig;         %Wp
            layerTwoOutput = layerTwoOutput + biasLayerTwo';      %+b = n
            layerTwoSig = logsig(layerTwoOutput);                 %a^2 = logsig(n)
            
            %output layer layerThreeOutput layerThreeLinear
            layerThreeOutput = outputMatrix* layerTwoSig;        %Wp
            layerThreeOutput = layerThreeOutput+biasOutputLayer';%+b = n
            layerThreeSig = logsig(layerThreeOutput);            %a^3 = logsig(n)

            
            % ERROR CALCULATION (target - output) 
            e = trainingLabels(:,r)-layerThreeSig;
            
            % Backpropagation
            
            %output sensitivity = -2F(N) * error
            outputSensitivity = -2*diag(dlogsig(layerThreeOutput, layerThreeSig))*e;
            
            
            %second layer sensitivity = layerTwoS
            layerTwoS = diag(dlogsig(layerTwoOutput,layerTwoSig))*(outputMatrix'*outputSensitivity);
            
            %first layer sensitivity = layerOneS
            layerOneS  = diag(dlogsig(layerOneOutput,layerOneSig))*(layerTwoMatrix'*layerTwoS);           
            
            %Update weights with calculated sensitivity
            outputMatrix  = outputMatrix -(learningRate.*outputSensitivity*layerTwoSig');
            biasOutputLayer = biasOutputLayer' - (learningRate*outputSensitivity);
            biasOutputLayer = biasOutputLayer';
            
            layerTwoMatrix = layerTwoMatrix -(learningRate.*layerTwoS*layerOneSig');          
            biasLayerTwo = biasLayerTwo' + (learningRate*layerTwoS);
            biasLayerTwo = biasLayerTwo';
            
            layerOneMatrix  = layerOneMatrix -(learningRate.*layerOneS*trainingImages(:,r)');
            biasLayerOne = biasLayerOne' + (learningRate*layerOneS);
            biasLayerOne = biasLayerOne';
            
            %Calculate the sum of errors squared
            sumError = (e'*e);
            
            %Add to vectors for later plotting
            E(t) = E(t) + sumError;
    end
    %Average the sum of errors squared for training data
    E(t) = E(t)/trainingPerEpoch;
    
    %run test after each epoch on test data
    for test = 1:size(testImages,2)
            layerOneOutput = layerOneMatrix * testImages(:,test); %WP
            layerOneOutput = layerOneOutput + biasLayerOne';      %+b = n
            layerOneSig = logsig(layerOneOutput);                 %a^1 = logsim(n)      
            
             %second layer layerTwoOutput
            layerTwoOutput =layerTwoMatrix * layerOneSig;         %Wp
            layerTwoOutput = layerTwoOutput + biasLayerTwo';      %+b = n
            layerTwoSig = logsig(layerTwoOutput);                 %a^2 = logsig(n)
            
            %output layer layerThreeOutput layerThreeLinear
            layerThreeOutput = outputMatrix* layerTwoSig;        %Wp
            layerThreeOutput = layerThreeOutput+biasOutputLayer';%+b = n
            layerThreeSig = logsig(layerThreeOutput);            %a^3 = logsig(n)

            
            % ERROR CALCULATION (target - output) 
            testError = testLabels(:,test)-layerThreeSig;
            
            %Calculate the sum of errors squared
            sumTestError = (testError'*testError);
            
            %Add to vectors for later plotting
            testE(t) = testE(t) + sumTestError;
    end
    
    %Average the sum of errors squared for test data
    testE(t) = testE(t)/size(testImages,2);
end
toc;



%Plot training error
figure;
plot(1:t,E,'b--o')
title('0.001 L Rate, 100 Neuron/Layer 100 Epoch Training Error Curve');
xlabel('Epochs');
ylabel('Squared Error Sum');

%Plot test error
figure;
plot(1:t,testE,'b--o')
title('0.001 L Rate, 100 Neuron/Layer 100 Epoch Test Error Curve');
xlabel('Epochs');
ylabel('Squared Error Sum');