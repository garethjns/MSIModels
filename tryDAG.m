%%
% Try creating a single channel models similar to the Keras models with
% MATLABs new LSTM layer, conv layer, and directed acyclic graphs.
% 


%% Conv model
% Create equivalent of ConvModels.simple()

close all force

% Parameters to worth with dataset 12 (see generateSets.m)
x1Width = 520;
nFil = 256;
ks = 128;

% Can't use sequence input with DAG (?), so what about a 1D image?
inp = imageInputLayer([1, x1Width], 'Name', 'input');
% No conv1d layer - use conv2D to do 1D
conv1 = convolution2dLayer([1, ks], nFil, ...
    'Stride', [1, 32], 'Name', 'conv_l1');
% No Flatten layer needed?
conv3 = dropoutLayer(0.3, 'Name', 'conv_l3');

% Create layers to make rate estimation
a1 = fullyConnectedLayer(x1Width/2, 'Name', 'rate_l1');
a2 = reluLayer('Name', 'rate_l2');
a3 = dropoutLayer(0.15, 'Name', 'rate_l3');
rateOutput1 = fullyConnectedLayer(1, 'Name', 'rate_l4');
rateOutput2 = reluLayer('Name', 'rate_l5');
rateOutput3 = regressionLayer('Name', 'rateOutput');

% Create layers to make final, "fast" or "slow" decision
decOutput1 = fullyConnectedLayer(2, 'Name', 'dec_l1');
decOutput2 = softmaxLayer('Name', 'dec_l2');
decOutput3 = classificationLayer('Name', 'decOutput');

% Add input layers
lGraph = layerGraph([inp, conv1, conv3]);
% Add rate layers
lGraph = addLayers(lGraph, ...
    [a1, a2, a3, rateOutput1, rateOutput2, rateOutput3]);
lGraph = connectLayers(lGraph, 'conv_l3', 'rate_l1');
% Add decision layers - can't add second output layer
lGraph = addLayers(lGraph, [decOutput1, decOutput2]);
lGraph = connectLayers(lGraph, 'rate_l5', 'dec_l1');

% Intended structure (minus decision output layer), although not actually
% used as has a lonely output
figure
plot(lGraph)


%% Load dataset 12
% Load dataset, prepare, and split into to train, valid, test.

close all force

% Load 
a = load('Data\stimData_AV_s12_20000x400.mat');

% Prepare
% Reshape from [n x length] to "image" [px, py, nChan, n]
soundsA = zeros(1, size(a.soundsA,2), 1, size(a.soundsA,1));
soundsA(1,:,1,:) = permute(a.soundsA, [2,1]);

% Generate decision labels, 1 = "fast"
decA = a.ratesA>10;
% One hot? Not needed?
decAOH = decA+1 == 1:max(decA+1);

% Split (dataset is already shuffled)
xTrain = soundsA(:,:,:,1:10000);
xValid = soundsA(:,:,:,10001:15000);
xTest = soundsA(:,:,:,15001:20000);
yTrain = categorical(decA(1:10000));
yValid = categorical(decA(10001:15000));
yTest = categorical(decA(15001:20000));
yTrainR = a.ratesA(1:10000);
yValidR = a.ratesA(10001:15000);
yTestR = a.ratesA(15001:20000);

% Just check reshape didn't do anyting unexpected
figure
idx = randi(10000);
plot(a.soundsA(idx,:))
hold on
plot(soundsA(1,:,1,idx))


%% Train Conv1D using rate as output
% Works

close all force

% Make graph with just input, conv, rate layers
lGraph = layerGraph([inp, conv1, conv3]);
lGraph = addLayers(lGraph, ...
    [a1, a2, a3, rateOutput1, rateOutput2, rateOutput3]);
lGraph = connectLayers(lGraph, 'conv_l3', 'rate_l1');

figure
plot(lGraph)

options = trainingOptions('sgdm', ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 1000,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {xValid, yValidR},...
    'Plots','training-progress');

% Train
net1 = trainNetwork(xTrain, yTrainR, lGraph, options);

% Predict
predTrain = net1.predict(xTrain);
predValid = net1.predict(xValid);
predTest = net1.predict(xTest);
disp(['Train loss: ', num2str(mse(single(predTrain), single(yTrainR)))])
disp(['Valid loss: ', num2str(mse(single(predValid), single(yValidR)))])
disp(['Test loss: ', num2str(mse(single(predTest), single(yTestR)))])


%% Try training with just decision as output
% Also works

close all force

% Make graph with inp, conv, rate, dec layers, where rate layers feed in to
% dec layers (rate output layers skipped)
lGraph = layerGraph([inp, conv1, conv3]);
lGraph = addLayers(lGraph, [a1, a2, a3]);
lGraph = connectLayers(lGraph, 'conv_l3', 'rate_l1');
lGraph = addLayers(lGraph, [decOutput1, decOutput2, decOutput3]);
lGraph = connectLayers(lGraph, 'rate_l3', 'dec_l1');

figure
plot(lGraph)

options = trainingOptions('sgdm', ...
    'MaxEpochs', 500, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', 1000, ...
    'InitialLearnRate', 0.01, ...
    'ValidationData', {xValid, yValid}, ...
    'Plots','training-progress');

% Train
net2 = trainNetwork(xTrain, yTrain, lGraph, options);

% Predict
predTrain = net2.predict(xTrain);
predValid = net2.predict(xValid);
predTest = net2.predict(xTest);
disp(['Train accuracy: ', ...
    num2str(sum((predTrain(:,2)>0.5) == double(yTrain)-1)/length(yTrain))])
disp(['Valid accuracy: ', ...
    num2str(sum((predValid(:,2)>0.5) == double(yValid)-1)/length(yValid))])
disp(['Test accuracy: ', ...
    num2str(sum((predTest(:,2)>0.5) == double(yTest)-1)/length(yTest))])



%% Train using decision as output, and including all rate layers 
% Not entirely sequential, not entirely working
% Error in tryDAG (line 185)
% net3 = trainNetwork(xTrain, yTrain, lGraph, options);
% 
% Caused by:
%     Error using nnet.cnn.LayerGraph>iInferSize (line 727)
%     Layer 9 is expected to have a different size.
    
close all force

% Make graph with inp, conv, rate, dec layers, where rate layers feed in to
% dec layers (final rate output layer skipped)

add = additionLayer(3, 'Name','add');

lGraph = layerGraph([inp, conv1, conv3]);
lGraph = addLayers(lGraph, [a1, a2, a3, rateOutput1, rateOutput2]);
lGraph = connectLayers(lGraph, 'conv_l3', 'rate_l1');
lGraph = addLayers(lGraph, [add, decOutput1, decOutput2, decOutput3]);
lGraph = connectLayers(lGraph, 'rate_l3', 'add/in1');
lGraph = connectLayers(lGraph, 'rate_l4', 'add/in2');
lGraph = connectLayers(lGraph, 'rate_l5', 'add/in3');

figure
plot(lGraph)

% Train
net3 = trainNetwork(xTrain, yTrain, lGraph, options);

% Predict
predTrain = net3.predict(xTrain);
predValid = net3.predict(xValid);
predTest = net3.predict(xTest);
disp(['Train accuracy: ', ...
    num2str(sum((predTrain(:,2)>0.5) == double(yTrain)-1)/length(yTrain))])
disp(['Valid accuracy: ', ...
    num2str(sum((predValid(:,2)>0.5) == double(yValid)-1)/length(yValid))])
disp(['Test accuracy: ', ...
    num2str(sum((predTest(:,2)>0.5) == double(yTest)-1)/length(yTest))])


%% LSTM
% trainSingleChannels.py uses LSTMModels.simple(), which has a fairly basic
% structure - 1 input, 2 outputs. Sequential except for first output.
% Problems:
% No sequence input or LSTM allowed in DAGs
% Only one output allowed

close all force

x1Width = 12;
inp = sequenceInputLayer(x1Width, 'Name', 'input');
lstm1 = lstmLayer(100, 'OutputMode', 'last', 'Name', 'LSTM_l1');
lstm2 = dropoutLayer(0.3, 'Name', 'LSTM_l3');

a1 = fullyConnectedLayer(x1Width/2, 'Name', 'rate_l1');
a2 = dropoutLayer(0.15, 'Name', 'rate_l2');
rateOutput1 = fullyConnectedLayer(1, 'Name', 'rate_l3');
rateOutput2 = reluLayer('Name', 'rateOutput');

decOutput1 = fullyConnectedLayer(2, 'Name', 'dec_l1');
decOutput2 = softmaxLayer('Name', 'decOutput');

try
    lgraph = layerGraph([inp, lstm1, lstm2]);
    figure
    plot(lgraph)
catch err
    disp(err)
end

try
    lgraph = layerGraph([lstm1, lstm2]);
    figure
    plot(lgraph)
catch err
    disp(err)
end

% :(
