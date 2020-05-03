# Deep multi-sensory integration models
![Tests](https://github.com/garethjns/MSIModels/workflows/Tests/badge.svg) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=garethjns_AudioDAG&metric=alert_status)](https://sonarcloud.io/dashboard?id=garethjns_AudioDAG)

Trying out early and late integration in multi-input neural networks, event detection and rate discrimination.

## Background
The human brain integrates incoming information over time, and combines evidence between different sensory modalities. Humans are able to use sensory information in a statistically-optimal fashion; reliable information is weighted as more important than unreliable or noisy information. However, exactly how and where the brain combines modalities is unclear, with multisensory processing occurring early in the cortex in cortices traditionally believed to be unisensory.

This project sets up analogies of two models of evidence integration - early and late - in ANNs and has two main aims:
 - To see how performance varies with "where" the integration occurs.
 - Compare the results to those observed in mammals, particularly with regard to 
    - The effect of temporal asynchrony between sensory modalities.
    - Which parts of the stimuli are most import for decision making.

### Early vs late integration
In early integration models, multisensory processing first occurs in sensory cortices (not necessarily exclusively). In late integration models, modalities are processed separately by the appropriate sensory cortices and then combined in later, decision making cortices (such as parietal and frontal) [[1](http://www.sciencedirect.com/science/article/pii/S0959438816300678)].  

Behavioural evidence implies late integration occurs [[2](http://www.jneurosci.org/content/32/11/3726.short)], anatomical evidence tends to imply integration occurs early and late.

## The task
Given noisy "sensory" input, the aim is to categorise the stimuli as "fast" or "slow", depending on the number of events in the input.

In the psychosocial version of this task, the appropriate fast/slow threshold is learned by the subject, and events are, for example, light flashes or tone pips that are embedded in a noisy background. 

For the neural networks, the events are the same, and the task is binary classification with a decision threshold set according to the mean of the known stim rates. In order to reach the final decision, each each individual component of the network performance sub-tasks such as event detection, rate estimation, etc.  These losses are also monitored.

## Stimuli
The stimuli used are based on those used in a number of psychophysical tasks (eg. [[2](http://www.jneurosci.org/content/32/11/3726.short), [3](http://www.eposters.net/poster/exploring-the-role-of-synchrony-in-auditory-visual-integration-in-ferrets-and-humans), [4](http://www.eposters.net/poster/exploring-the-role-of-synchrony-in-spatial-and-temporal-auditory-visual-integration-1)]). The stimuli are either "unisensory" (single channel input) or "multisensory" (two-channel input).   


Each channel consists a number of events embedded in a noisy background. These events are separated by gaps of two discrete lengths. For comparison, in a auditory and visual psycphysics task, these inputs are analagous to the raw voltage traces used to drive speakers (AC) and LEDs (rectified). In this case, the
event might be a short flash (~20 ms) for the visual channel, and a a short tone pip (~20 ms) for the auditory channel.

In the multisensory case, the two channels can be:
 - Matched: The same number of events on each channel
   - Asynchronous: Matched, but events between channel temporally aligned between channels
   - Synchronous: Matched AND the events align temporally between channels
 - Unmatched: 
   - Agreeing: Differing number of events on each channel, but indicating the same decision (ie. both "fast" or both "slow")
   - Conflicting: Differing number of events on each channel, and indicating contradictory decisions (ie. one "fast, one "slow")

![Example stim](https://github.com/garethjns/MSIModels/blob/master/images/stimExample.png) 

## Models
The models are developed in Keras and have two inputs and multiple outputs. The inputs take the raw time series inputs and use 1D convolution layers.  
 
There outputs are outputs for the convolution layers for each modality, and later fully connected layers for rate and finally for the "fast" or "slow" decision.

Modifying the weights of the outputs allows creation of models that either on the final decision, or the event detection stages.
### Early
![Example stim](https://github.com/garethjns/MSIModels/blob/master/images/mod_early.png) 
### Intermediate
![Example stim](https://github.com/garethjns/MSIModels/blob/master/images/mod_intermediate.png) 
### Late
![Example stim](https://github.com/garethjns/MSIModels/blob/master/images/mod_late.png) 
# Install

```bash
git clone https://github.com/garethjns/MSIModels.git
cd MSIModels
pip install .
```

# Running

## Data preparation
Individual stimuli are constructed with [AudioDAG](https://github.com/garethjns/AudioDAG) and combined into datasets using msi_models.stimset.channel and msi_models.stimset.multi_channel objects. The handle feeding data from disk from pre-generated batches, or on the fly. It's generally better to use pre-generated datasets as the generation is fairly slow and can bottleneck training. The classes include methods for generating these batches.

Templates are available for different stimulus components, types and combinations of stimuli. The typically specify compatible defaults for the params, which can be modified.

This creates a .hdf5 file with the following structure:
```
    |--left/
            |--x (n, duration_pts, 1)
            |--x_indicators (n, duration_pts, 1)
            |--y_rate (n,)
            |--y_dec (n,)
            |--configs 
    |--right/
            |--x (n, duration_pts, 1)
            |--x_indicators (n, duration_pts, 1)
            |--y_rate (n,)
            |--y_dec (n,)
            |--configs 
    /--agg/
           |--y_rate
           |--y_dec
```

Generating multisensory (two channel) using templates.

```python
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_templates import template_sync

MultiTwoGapStim.generate(template=template_sync,
                         fs=500,
                         n=50,
                         batch_size=10,
                         fn='data/sample_multisensory_data_sync.hdf5',
                         n_jobs=-1,
                         template_kwargs={"duration": 1300,
                                          "background_mag": 0.09,
                                          "duration_tol": 0.5})
```

Additional examples:
scripts/generate_multisensory_data.py  
scripts/generate_unisensory_data.py  

## Single model fitting

Model templates are available in msi_models.models.conv.multisensory_templates, which define combinations of loss weights to create a binary classifier, event detector, etc. MultisensoryClassifier creates a model aims to correctly predict rate and the "fast"/"slow" decision.

The MultisensoryClassifier uses the agg/y_rate and agg/y_dec as output, targets, but also calculates the loss for the against the unisensory targets, eg left/y_rate and left/y_dec so the performance of the individual outputs can also be monitored.

The channels are first defined, then combined into a MultiChannel object to feed the model.

Using previously generated data (see above or use the small example sets in data/, although these aren't big enough to learn anything useful):
````python
import os
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

# Prepare data feeder
fn = "data/sample_multisensory_data_matched.hdf5"
path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')
common_kwargs = {"path": path,
                 "train_prop": 0.8,
                 "x_keys": ["x", "x_mask"],
                 "y_keys": ["y_rate", "y_dec"],
                 "seed": 100}

left_config = ChannelConfig(key='left', **common_kwargs)
right_config = ChannelConfig(key='right', **common_kwargs)
multi_config = MultiChannelConfig(path=path,
                                  key='agg',
                                  y_keys=["y_rate", "y_dec"],
                                  channels=[left_config, right_config])

mc = MultiChannel(multi_config)

# View examples of the prepared data
mc.plot_example()
mc.plot_example()

# Create and fit the model
mod = MultisensoryClassifier(integration_type='intermediate_integration',
                             opt='adam',
                             epochs=1000,
                             batch_size=2000,
                             lr=0.0025)

mod.fit(mc.x_train, mc.y_train,
        validation_split=0.4,
        epochs=1000)
````
Additional examples:
scripts/train_unisensory_model.py
scripts/train_multisensory_model.py

## Running experiments

### ExperimentalRuns
An ExperimentalRun is a component of [Experiment](#full_experiment) designed to run the same model on the same data set multiple times (with different seeds), and adds logging and performance statistics. The data and model are constructed as usual, then wrapped as ExperimentalDatasets and ExperimentalModels. The classes handle the stochastic elements for each run (shuffling training sets, model weight instantiation, etc.)

See [View results](#view_results) section below to setup up MLflow to view the experimental logs.

scripts/run_single_experiment.py

```python
import os
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig

fn = "data/sample_multisensory_data_matched.hdf5"
path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

common_kwargs = {"path": path,
                 "train_prop": 0.8,
                 "x_keys": ["x", "x_mask"],
                 "y_keys": ["y_rate", "y_dec"],
                 "seed": 100}

left_config = ChannelConfig(key='left', **common_kwargs)
right_config = ChannelConfig(key='right', **common_kwargs)
multi_config = MultiChannelConfig(path=path,
                                  key='agg',
                                  y_keys=["y_rate", "y_dec"],
                                  channels=[left_config, right_config])

exp_data = ExperimentalDataset(name='multi_matched',
                               config=multi_config)
exp_data.build(seed=123)

# Prepare model
mod = MultisensoryClassifier(integration_type='intermediate_integration',
                             opt='adam',
                             epochs=1000,
                             batch_size=500,
                             lr=0.0025)
exp_model = ExperimentalModel(model=mod,
                              name="multi_inter")

# Prepare exp run
exp_run = ExperimentalRun(data=exp_data,
                          model=exp_model,
                          n_reps=4)

exp_run.run()
```
<a name="full_experiment"></a>
### Full Experiment
The Experiment class handles running multiple ExperimentalRuns on different ExperimentaModel and ExperimentaDataset combinations, and additional stats and logging.

In the MLflow logs, two experiments are created - one with the supplied name and the second with the suffix _agg. The experiment with the supplied name includes the n ExperimentalRun outputs as runs (rows)., The _agg version reduces these to aggregate Model + Data combinations and appropriate stats.

```python
from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig

# Prepare runs
common_data_kwargs = {"train_prop": 0.8,
                      "x_keys": ["x", "x_mask"],
                      "y_keys": ["y_rate", "y_dec"],
                      "seed": 100}
path='data'
fns = {"multi_sync": "sample_multisensory_data_sync.hdf5",
       "multi_matched": "sample_multisensory_data_matched.hdf5",
       "multi_unmatched": "sample_multisensory_data_unmatched.hdf5"}

exp_datasets = []
for fk, fn in fns.items():
    exp_datasets.append(
        ExperimentalDataset(name=fk,
                            config=MultiChannelConfig(path=f"{path}/{fn}",
                                                      key='agg',
                                                      y_keys=["y_rate", "y_dec"],
                                                      channels=[ChannelConfig(path=f"{path}/{fn}",
                                                                              key='left', **common_data_kwargs),
                                                                ChannelConfig(path=f"{path}/{fn}",
                                                                              key='right', **common_data_kwargs)])))

exp_models = []
for mod in ['early_integration', 'intermediate_integration', 'late_integration']:
    exp_models.append(
        ExperimentalModel(model=MultisensoryClassifier(integration_type='intermediate_integration',
                                                       opt='adam',
                                                       epochs=10,
                                                       batch_size=500,
                                                       lr=0.0025)))

experiment = Experiment(models=exp_models,
                        datasets=exp_datasets,
                        name="example_experiment2")

experiment.run()
```

See [View results](#view_results) section below to setup up MLflow to view the experimental logs.

scripts/run_full_experiment.py

<a name="view_results"></a>
## View results

1) Run MLflow server. Use a Linux VM or WSL if running project in Windows.

````bash
cd path/containing/mlruns
mlflow server
````

2) View at http://localhost:5000


# TODO:
 - Current models are prototypes and vary wildly in the total number of trainable parameters.
 - Model eval - SHAP
  
