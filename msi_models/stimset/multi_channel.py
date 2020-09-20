import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msi_models.stimset.channel import Channel
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel_config import MultiChannelConfig

os.sep = '/'


class MultiChannel:
    def __init__(self, config: MultiChannelConfig):
        self.config = config
        # Create channels to get individual channel x and y data
        self.channels = [Channel(config) for config in self.config.channels]
        # Create a new channel to get the main y keys, wherever they are.
        self.y_channel = Channel(ChannelConfig(path=self.config.path,
                                               seed=self.config.seed,
                                               key=self.config.key,
                                               y_keys=self.config.y_keys))

        self.n = self.channels[0].n

        self.y_keys = [os.path.join(self.config.key, k).replace('\\', '/') for k in self.config.y_keys]

        self.summary: pd.DataFrame = pd.read_hdf(self.config.path, key='summary', mode='r')
        self.summary_train = self.summary.iloc[self.channels[0].train_idx]
        self.summary_test = self.summary.iloc[self.channels[0].test_idx]

    @property
    def x(self):
        x = {}
        for c in self.channels:
            x.update(c.x)
        return x

    @property
    def y(self):
        ys = {}
        for c in [self.y_channel] + self.channels:
            ys.update(c.y)

        return ys

    @property
    def x_train(self):
        x = {}
        for c in self.channels:
            x.update(c.x_train)
        return x

    @property
    def x_test(self):
        x = {}
        for c in self.channels:
            x.update(c.x_test)
        return x

    @property
    def y_train(self):
        ys = {}
        for c in [self.y_channel] + self.channels:
            ys.update(c.y_train)

        return ys

    @property
    def y_test(self):
        ys = {}
        for c in [self.y_channel] + self.channels:
            ys.update(c.y_test)

        return ys

    def plot_example(self, show: bool = True):
        row = np.random.choice(range(0, self.n))
        for k, v in self.y.items():
            print(f"{k}: {v[row]}")

        fig, axs = plt.subplots(nrows=2, ncols=1)
        for i, (c, ax) in enumerate(zip(["left", "right"], axs)):
            ax.plot(self.x[f'{c}_x'][row], label=f'Signal (x)')
            ax.plot(self.x[f'{c}_x_mask'][row], label=f'Events mask (y)')
            ax.set_title(f'"{c.capitalize()}" channel, rate: {self.y[f"{c}_y_rate"][row]}, '
                         f'decision: {self.y[f"{c}_y_dec"][row]}', fontweight='bold')
            ax.set_ylabel('Mag', fontweight='bold')
            if i == 1:
                ax.set_xlabel('Time', fontweight='bold')
                ax.legend(loc='lower right')

        fig.suptitle(f"Type: {self.summary.loc[row, 'type']}, aggregated rate: {self.y['agg_y_rate'][row]}, "
                     f"decision {self.y['agg_y_dec'][row]}", fontweight='bold')
        fig.tight_layout()

        if show:
            plt.show()


if __name__ == "__main__":
    path = "../../data/sample_multisensory_data_matched_250k.hdf5"
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

    mc.x
    mc.x_train.keys()
