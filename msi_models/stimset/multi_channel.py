import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from msi_models.stimset.channel import Channel
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel_config import MultiChannelConfig


class MultiChannel:
    def __init__(self, config: MultiChannelConfig):
        self.config = config
        # Create channels to get individual channel x and y data
        self.channels = [Channel(config) for config in self.config.channels]
        # Create a new channel to get the main y keys, wherever they are.
        self.y_channel = Channel(ChannelConfig(path=self.config.path, seed=self.config.seed,
                                               key=self.config.key, y_keys=self.config.y_keys,
                                               train_prop=self.config.channels[0].train_prop))

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

    def plot_example(self, show: bool = True) -> plt.Figure:
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

        return fig

    def plot_summary(self, subset='all', show: bool = True) -> plt.Figure:

        if subset == 'all':
            data = self.summary
        elif subset == 'train':
            data = self.summary_train
        elif subset == 'test':
            data = self.summary_test
        else:
            raise ValueError(f"Invalid subset {subset}, use 'all, 'train' or 'test.")

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2)

        for ax_i, side in enumerate(['left', 'right']):
            ax = fig.add_subplot(gs[0, ax_i])
            rates = [r for r in data[f"{side}_n_events"].unique() if r != 0]
            sns.histplot(data.loc[data[f"{side}_n_events"] != 0, :], y=f"{side}_n_events", hue="type", ax=ax,
                         bins=len(rates))
            ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            ax.set_title(side.capitalize(), fontweight='bold')
            if ax_i == 0:
                ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
            else:
                ax.set_ylabel('')

        ax = fig.add_subplot(gs[1, :])
        sns.histplot(data.type, label=type, ax=ax, kde=False, bins=len(data.type.unique()))
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
        fig.suptitle(f"{os.path.split(self.config.path)[-1]}, subset: {subset.capitalize()}", fontweight='bold')

        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    path = "../../scripts/data/sample_multisensory_data_mix_hard_250k.hdf5"
    common_kwargs = {"path": path, "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"], "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}

    multi_config = MultiChannelConfig(path=path, key='agg', y_keys=["y_rate", "y_dec"],
                                      channels=[ChannelConfig(key='left', **common_kwargs),
                                                ChannelConfig(key='right', **common_kwargs)])

    mc = MultiChannel(multi_config)

    mc.plot_example()

    mc.x.keys()
    mc.x_train.keys()

    mc.plot_summary()
