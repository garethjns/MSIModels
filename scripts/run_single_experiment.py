"""Example running repeated model + data experiment, and logging, using ExperimentalRun"""

import os

import tensorflow as tf

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=5000)])

N_REPS = 5
N_EPOCHS = 2

if __name__ == "__main__":
    # Prepare data
    fn = "data/sample_multisensory_data_mix_hard_250k.hdf5"
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    common_kwargs = {"path": path, "train_prop": 0.8, "seed": 100,
                     "x_keys": ["x", "x_mask"], "y_keys": ["y_rate", "y_dec"]}

    multi_config = MultiChannelConfig(path=path, key='agg', y_keys=common_kwargs["y_keys"],
                                      channels=[ChannelConfig(key='left', **common_kwargs),
                                                ChannelConfig(key='right', **common_kwargs)])
    data = MultiChannel(multi_config)

    # Prepare model
    mod = ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                   opt='adam', batch_size=2000, lr=0.01),
                            name='example_model')

    # Prepare run
    exp_run = ExperimentalRun(name=f"example_run_for_example_model", model=mod, data=data,
                              n_reps=N_REPS, n_epochs=N_EPOCHS)

    # Run
    exp_run.run()

    # Evaluate
    exp_run.evaluate()

    # View results
    exp_run.log_run(to='example_run_summary')
    exp_run.log_summary(to='example_run')
    exp_run.results.plot_aggregated_results()
    print(exp_run.results.curves_agg)
