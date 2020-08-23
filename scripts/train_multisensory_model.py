"""Example training multisensory model."""

import os

import tensorflow as tf

from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=10000)])

if __name__ == "__main__":
    fn =  '../data/sample_multisensory_data_unmatched_250k.hdf5'
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

    mc.plot_example()
    mc.plot_example()

    mod = MultisensoryClassifier(integration_type='intermediate_integration',
                                 opt='adam',
                                 batch_size=500,
                                 lr=0.004)

    y_names = ['agg_y_rate', 'agg_y_dec', 'left_conv_1', 'left_flatten_1', 'right_conv_1',
               'right_flatten_1', 'left_rate_output', 'right_rate_output']
    mod.fit(mc.x_train, {k: v for k, v in mc.y_train.items() if k in y_names},
            epochs=1200,
            validation_split=0.4)
