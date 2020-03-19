import os
from datetime import datetime
from typing import List, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, root_validator, PositiveInt, validator, FilePath

from msi_models.exceptions.params import IncompatibleParametersException, InvalidParameterException
from msi_models.stimset.channel import Channel
from msi_models.stimset.channel import ChannelConfig


os.sep = '/'


class MultiChannelConfig(BaseModel):
    """
    Config for multiple input channel models, with 2 main outputs.

    :param path: Path to hdf5 file containing y_keys.
    :param key: Root key to use to access y_keys. eg. 'agg', 'left', 'right'.
    :param y_keys: The keys for the aggregate/overall y_rate and y_dec that will be used for training the model.
                   Suffixed to key param.
    :param channels: List of ChannelConfigs specifying the x_data for each channel, and also the single channel y_data
                     (if needed).
    :param seed: Int specifying a numpy seed. All channels will be set to this seed so that train/test splitting will be
                 consistent.
    """
    path: FilePath
    key: str = 'agg/'
    y_keys: List[str]
    channels: List[ChannelConfig]
    seed: Union[PositiveInt] = None

    @validator('seed', pre=True)
    def check_seed(cls, v: Union[int, None]) -> int:
        # This doesn't run?
        if v is None:
            return int(datetime.now().timestamp())
        else:
            return v

    @root_validator
    def all_train_props_match(cls, values):
        if len(np.unique([float(c.train_prop) for c in values["channels"]])) > 1:
            raise IncompatibleParametersException(f"Train props not consistent between specified channels.")

        return values

    @root_validator
    def all_seeds_match(cls, values):
        seed = values.get("seed", None)
        if seed is None:
            values["seed"] = int(datetime.now().timestamp())

        for c in values["channels"]:
            c.seed = values["seed"]

        return values

    @root_validator
    def y_keys_exist_and_match_len(cls, values):
        """Check file path's aggregate y values."""
        with h5py.File(values['path'], 'r') as f:
            key = "/" if values["key"] == "" else values["key"]
            keys = list(f[key].keys())

        if not np.all([v in keys for v in values["y_keys"]]):
            raise InvalidParameterException(f"Some of y_keys ({values['y_keys']}) missing from file keys ({keys})")

        return values


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

        self.y_keys = [os.path.join(self.config.key, k) for k in self.config.y_keys]

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
    common_kwargs = {"path": "data/multisensory_data.hdf5",
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}

    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)

    multi_config = MultiChannelConfig(channels=[left_config, right_config])

    mc = MultiChannel(multi_config)

    mc.plot_example()

    mc.x
    mc.x_train.keys()
