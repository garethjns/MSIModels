"""Example training a model using the experimental wrappers"""
import os
from typing import List

import tensorflow as tf

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=5000)])

if __name__ == "__main__":
    # Prepare data
    fn = 'data/sample_multisensory_data_mix_hard_250k.hdf5'
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    common_kwargs = {"path": path,
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}

    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)
    multi_config = MultiChannelConfig(path=path, key='agg',
                                      y_keys=common_kwargs["y_keys"],
                                      channels=[left_config, right_config])
    mc = MultiChannel(multi_config)

    # Prepare models
    common_model_kwargs = {'opt': 'adam',
                           'batch_size': 15000,
                           'lr': 0.007}
    early_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='early_integration',
                                                               **common_model_kwargs))
    int_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                             **common_model_kwargs))
    late_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='late_integration',
                                                              **common_model_kwargs))

    for mod in [early_exp_model, int_exp_model, late_exp_model]:
        # Fit
        mod.fit(mc, epochs=1000)
        # Eval
        # mod.plot_example(mc, dec_key='agg_y_dec')
        mod.calc_prop_fasts(mc, rate_key='agg_y_rate')
        _, test_psyche_fits = mod.calc_psyche_curves(mc, rate_key='agg_y_rate')
        print(test_psyche_fits)
        mod.plot_prop_fast(mc, rate_key='agg_y_rate')
        train_report, test_report = mod.report(mc)


