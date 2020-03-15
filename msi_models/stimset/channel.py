from typing import List, Union, Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, FilePath, root_validator

from msi_models.exceptions.params import InvalidParameterException


class ChannelConfig(BaseModel):
    path: FilePath
    x_keys: List[str]
    y_keys: List[str]
    seed: int = 0
    train_prop: float = 0.8

    @root_validator
    def all_keys_exist_and_match_len(cls, values):
        with h5py.File(values['path'], 'r') as f:
            keys = list(f.keys())

        if not np.all([v in keys for v in values["x_keys"]]):
            raise InvalidParameterException(f"Some of x_keys ({values['x_keys']}) missing from file keys ({keys})")

        if not np.all([v in keys for v in values["y_keys"]]):
            raise InvalidParameterException(f"Some of y_keys ({values['y_keys']}) missing from file keys ({keys})")

        return values


class Channel:
    def __init__(self, channel_config: ChannelConfig):
        self.config = channel_config
        with h5py.File(self.config.path, 'r') as f:
            self.n: int = f[channel_config.y_keys[0]].shape[0]

        self._x: Dict[str, np.ndarray] = None
        self._y: Dict[str, np.ndarray] = None
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None

    def _load(self, keys: Union[List[str], str]) -> Dict[str, np.ndarray]:
        if not isinstance(keys, list):
            keys = [keys]

        with h5py.File(self.config.path, 'r') as f:
            data = {k: f[k][:] for k in keys}

        return data

    def _split(self):
        if self.train_idx is None:
            self.n_train = int(self.n * self.config.train_prop)
            self.n_test = self.n - self.n_train
            np.random.RandomState(self.config.seed)
            shuffled_idx = np.random.choice(range(self.n),
                                            replace=False,
                                            size=self.n)
            self.train_idx = shuffled_idx[0: self.n_train]
            self.test_idx = shuffled_idx[self.n_train::]

    @property
    def x(self) -> Dict[str, np.ndarray]:
        if self._x is None:
            x = self._load(keys=self.config.x_keys)
            self._x = x
            self.n = list(x.values())[0].shape[0]

        return self._x

    @property
    def y(self) -> Dict[str, np.ndarray]:
        if self._y is None:
            y = self._load(keys=self.config.y_keys)
            self._y = self._load(keys=self.config.y_keys)
            self.n = list(y.values())[0].shape[0]

        return self._y

    @property
    def x_train(self):
        self._split()
        return {k: v[self.train_idx] for k, v in self.x.items()}

    @property
    def x_test(self):
        self._split()
        return {k: v[self.test_idx] for k, v in self.x.items()}

    @property
    def y_train(self):
        self._split()
        return {k: v[self.train_idx] for k, v in self.y.items()}

    @property
    def y_test(self):
        self._split()
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
