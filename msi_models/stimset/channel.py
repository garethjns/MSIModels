import os
from typing import List, Dict, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from msi_models.stimset.channel_config import ChannelConfig


class Channel:
    def __init__(self, channel_config: ChannelConfig):
        self.config = channel_config
        self.x_keys = [os.path.join(self.config.key, k).replace('\\', '/') for k in self.config.x_keys]
        self.y_keys = [os.path.join(self.config.key, k).replace('\\', '/') for k in self.config.y_keys]

        with h5py.File(self.config.path, 'r') as f:
            self.n: int = f[self.y_keys[0]].shape[0]

        self._x: Union[None, Dict[str, np.ndarray]] = None
        self._y: Union[None, Dict[str, np.ndarray]] = None
        self.train_idx: Union[None, np.ndarray] = None
        self.test_idx: Union[None, np.ndarray] = None

        self._split()

    def _load(self, keys: Union[List[str], str]) -> Dict[str, np.ndarray]:
        if not isinstance(keys, list):
            keys = [keys]

        with h5py.File(self.config.path, 'r') as f:
            data = {k.replace("/", "_"): f[k][:] for k in keys}

        return data

    def _split(self):
        if self.train_idx is None:
            self.n_train = max(int(self.n * self.config.train_prop), 1)
            self.n_test = self.n - self.n_train

            state = np.random.RandomState(self.config.seed)
            shuffled_idx = np.array(range(self.n))
            state.shuffle(shuffled_idx)

            self.train_idx = shuffled_idx[0: self.n_train]
            self.test_idx = shuffled_idx[self.n_train::]

    @property
    def x(self) -> Dict[str, np.ndarray]:
        if self._x is None:
            x = self._load(keys=self.x_keys)
            self._x = x
            self.n = list(x.values())[0].shape[0]

        return self._x

    @property
    def y(self) -> Dict[str, np.ndarray]:
        if self._y is None:
            y = self._load(keys=self.y_keys)
            self._y = y
            self.n = list(y.values())[0].shape[0]

        return self._y

    @property
    def x_train(self):
        return {k: v[self.train_idx] for k, v in self.x.items()}

    @property
    def x_test(self):
        return {k: v[self.test_idx] for k, v in self.x.items()}

    @property
    def y_train(self):
        return {k: v[self.train_idx] for k, v in self.y.items()}

    @property
    def y_test(self):
        return {k: v[self.test_idx] for k, v in self.y.items()}

    def plot_example(self,
                     show: bool = True):
        row = np.random.choice(range(0, self.n))
        for k, v in self.y.items():
            print(f"{k}: {v[row]}")

        for v in self.x.values():
            plt.plot(v[row])

        if show:
            plt.show()


if __name__ == "__main__":
    chan_config = ChannelConfig(path='data/unisensory_data.hdf5',
                                x_keys=['x_1', 'x_indicators'],
                                y_keys=['y_rate', 'y_dec'])

    chan = Channel(chan_config)

    chan.x_train
    chan.plot_example()
