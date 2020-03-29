"""Example training unisensory model."""

import os

from msi_models.models.conv.unisensory_templates import UnisensoryClassifier
from msi_models.stimset.channel import ChannelConfig, Channel

if __name__ == "__main__":
    fn = 'data/unisensory_data_hard.hdf5'
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    chan_config = ChannelConfig(path=path,
                                x_keys=['x_1', 'x_indicators'],
                                y_keys=['rate_output', 'dec_output'])

    chan = Channel(chan_config)

    chan.plot_example(show=True)
    chan.plot_example(show=True)

    mod = UnisensoryClassifier(opt='adam',
                               epochs=1000,
                               batch_size=2500,
                               lr=0.0025)

    mod.fit(chan.x_train, chan.y_train,
            validation_split=0.4,
            epochs=1000)
