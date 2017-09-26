# Deep multi-sensory integration models
## Aims
 - Create artificial neural networks (ANNs) that are able to detect events in two channel time-series and perform a rate-discrimination task.
 - Compare performance of ANNs using LSTMs and 1D Convolutions for event detection.
 - Compare performance between networks where information in each time-series is combined in early and late layers, as an analogy for "early" and "late" integration models in the brain
 - Compare robustness of simple and more complex ANNs to signal noise.
 - Try out MATLAB 2017b's LSTM layer and directed acyclic graphs (DAGs)


## Running
1) Generate a dataset using **generateSets.m**.
2) Run **trainSingleChannels.py** to train and test LSTM and Conv1D models constructed in Keras.
3) Run **tryDAG.m** to try out equivalent models in MATLAB.

## Requirements
 - MATLAB 2014b+ to run **generateSets.m**.
 - MATLAB 2017b and Neural Network toolbox to run **tryDAG.m**.
 - Python 3.5, Keras, TensorFlow to run **trainSingleChannels.py**.

## Background
The human brain integrates incoming information over time, and combines evidence between different sensory modalities. Humans are able to use sensory information in a statistically-optimal fashion; reliable information is weighted as more important than unreliable or noisy information. However, exactly how and where the brain combines modalities is unclear, with multisensory processing occurring early in the cortex in cortices traditionally believed to be unisensory.

This project aims to set up analogies of two models of evidence integration - early and late - in ANNs to see how performance varies with "where" the integration occurs.

### Early vs late integration
In early integration models, multisensory processing first occurs in sensory cortices (not necessarily exclusively). In late integration models, modalities are processed separately by the appropriate sensory cortices and then combined in later, decision making cortices (such as parietal and frontal) [[1]](http://www.sciencedirect.com/science/article/pii/S0959438816300678).  

Behavioural evidence implies late integration occurs [[2]](http://www.jneurosci.org/content/32/11/3726.short), anatomical evidence tends to imply integration occurs early and late.


## Data/stimuli
The stimuli used here are the same as used in a number of psychophysical tasks. They consist of an auditory and visual channel, each of which are around 1 second long and contain "events" embedded in a noisy background.

For visual channel, an event is a short flash (20 ms), for the auditory channel, a short tone pip (20 ms).

Currently, the number of events is the same for the A and V channels, but the distribution of events can differ.

![Example stim](https://github.com/garethjns/MSIModels/blob/master/Images/stimExample.png) 

## Task
The task of the subject (human or ANN) is to classify a given stimulus - which could be auditory only, visual only, or AV as "fast" or "slow".

To perform the task optimally, a subject has to be able to detect events in each channel, estimate the rate (which requires memory), and combine information between the streams weighted by reliability.

Humans learn the discrimination threshold from feedback, the ANNs can learn from labelled data.

# Artificial neural networks
A number of models are planned for be implementation here:
 - Single channel, simple, LSTM
	- See **LSTMModels.simple()**
	- First layer is single LSTM
	- Inputs: Single time-series
 	- Outputs: Estimate of event rate; "fast" or "slow" decision. 
 - Single channel, simple, 1D conv
	- See **ConvModels.simple()**
	- First layer is single 1D convolution
	- Inputs: Single time-series
 	- Outputs: Estimate of event rate; "fast" or "slow" decision. 
 - Single channel, complex
	- Not yet implemented
	- More layers; sequential convolution/LSTM layers
	- Inputs: Single time-series
 	- Outputs: Encoded time-series; estimate of event rate; "fast" or "slow" decision. 
  - Multi channel, early integration
	- LSTM and/or conv1D for first layers
	- AV combination in early layers
	- Inputs: Auditory time-series; visual time-series
	- Outputs:estimate of event rate; "fast" or "slow" decision. 
  - Multi channel, late integration
	- As above, but AV combination in later layers (after unisensory rate estimation)

## Single channel models
### LSTM (Keras)
````PYTHON
# Prepare inputs
x1Width = x1.shape[1] # Also aud lstm output width
        
# Create Input layers
inp = Input(shape=(x1Width,1), dtype='float32', name='input')
        
# Aud LSTM    
lstm = LSTM(nDims, input_shape=(x1Width,1), 
               return_sequences=True, name='LSTM_l1')(inp)
lstm = Flatten(name='LSTM_l2')(lstm) 
lstm = Dropout(0.3, name='LSTM_l3')(lstm) 
        
# Aud dense layers
a = Dense(int(x1Width/2), activation='relu', 
           name='rate_l1')(lstm)
a = Dropout(0.15, name='rate_l2')(a)
rateOutput = Dense(1, activation='relu', 
                      name='rateOutput')(a)
        
# And make decision
decOutput = Dense(2, activation='softmax', 
                     name='decOutput')(rateOutput)
        
# Make model with 1 input and 2 outputs
model = Model(inputs=[inp],
              outputs=[rateOutput, decOutput])
        
# Compile with weighted losses
model.compile(optimizer='rmsprop', loss='mse',
               loss_weights=[0.5, 0.5], 
               metrics=['accuracy'])
      
````

### Conv1D (Keras)
````
 # Prepare inputs
 x1Width = x1.shape[1] # Also aud lstm output width
      
 # Create Input layers
 inp = Input(shape=(x1Width,1), dtype='float32', name='input')
        
 # Aud LSTM    
 conv = Conv1D(nFil, kernel_size= ks, strides=strides, 
                  input_shape=(x1Width,1), 
                  name='audConv_l1')(inp)
 conv = Flatten(name='audConv_l2')(conv) 
 conv = Dropout(0.3, name='audConv_l3')(conv) 
        
 # Aud dense layers
 a = Dense(int(x1Width/2), activation='relu', 
           name='rate_l1')(conv)
 a = Dropout(0.15, name='rate_l2')(a)
 rateOutput = Dense(1, activation='relu', 
                       name='rateOutput')(a)
        
 # And make decision
 decOutput = Dense(2, activation='softmax', 
                       name='decOutput')(rateOutput)
        
 # Make model with 1 input and 2 outputs
 model = Model(inputs=[inp],
                outputs=[rateOutput, decOutput])
        
 # Complile with weighted losses
 rmsprop = optimizers.rmsprop(lr=0.0001)
 model.compile(optimizer=rmsprop, loss='mse',
               loss_weights=[0.5, 0.5], 
                metrics=['accuracy'])
````

### Conv1D (MATLAB)
````MATLAB
inp = imageInputLayer([1, x1Width], 'Name', 'input');
conv1 = convolution2dLayer([1, ks], nFil, ...
    'Stride', [1, 32], 'Name', 'conv_l1');
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
````

![DAG](https://github.com/garethjns/MSIModels/blob/master/Images/conv1DDAG.png) 

# Multi-channel models
- WIP