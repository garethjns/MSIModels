"""Example training unisensory model."""

import os

import shap
import tensorflow as tf

from msi_models.models.conv.unisensory_templates import UnisensoryClassifier
from msi_models.stimset.channel import Channel
from msi_models.stimset.channel_config import ChannelConfig

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=6000)])

if __name__ == "__main__":
    fn = 'data/sample_unisensory_data_hard.hdf5'
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    chan_config = ChannelConfig(path=path,
                                x_keys=['x', 'x_mask'],
                                y_keys=['y_rate', 'y_dec'])

    chan = Channel(chan_config)

    chan.plot_example(show=True)
    chan.plot_example(show=True)

    mod = UnisensoryClassifier(opt='adam',
                               epochs=1000,
                               batch_size=1000,
                               lr=0.0025)

    mod.fit(chan.x_train, chan.y_train,
            validation_split=0.4,
            epochs=10)

    # Check predictions for first 10 in test set
    y_pred = mod.predict_dict(chan.x_test['x'][0:10, :])
    for i in range(10):
        print(f"Rate: {chan.y_test['y_rate'][i]} <-> {y_pred['y_rate'][i]} "
              f"| Decision: {chan.y_test['y_dec'][i]} <-> {y_pred['y_dec'][i]}")
