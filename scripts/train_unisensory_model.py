"""Example training unisensory model."""

import os

from msi_models.models.conv.unisensory_templates import UnisensoryClassifier
from msi_models.stimset.channel import Channel
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.tf_helpers import limit_gpu_memory

if __name__ == "__main__":
    limit_gpu_memory(5000)

    # Specify a dataset generated by scripts/generate_unisensory_data.py
    path = 'data/scripts_unisensory_data_hard.hdf5'
    if not os.path.exists(path):
        raise RuntimeWarning(f"Need to generate data first - run scripts/generate_unisensory_data.py")

    chan_config = ChannelConfig(path=path, x_keys=['x', 'x_mask'], y_keys=['y_rate', 'y_dec'])
    chan = Channel(chan_config)

    # plot some data examples
    chan.plot_example(show=True)
    chan.plot_example(show=True)

    # Prepare and fit mod
    mod = UnisensoryClassifier(opt='adam', epochs=1000, batch_size=1000, lr=0.0025)
    mod.fit(chan.x_train, chan.y_train, validation_split=0.4, epochs=10)

    # Check predictions for first 10 in test set
    y_pred = mod.predict_dict(chan.x_test['x'][0:10, :])
    for i in range(10):
        print(f"Rate: {chan.y_test['y_rate'][i]} <-> {y_pred['y_rate'][i]} "
              f"| Decision: {chan.y_test['y_dec'][i]} <-> {y_pred['y_dec'][i]}")
