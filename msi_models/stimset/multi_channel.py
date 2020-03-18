import warnings
from typing import List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, root_validator, PositiveInt, validator

from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stimset.channel import Channel
from msi_models.stimset.channel import ChannelConfig
from datetime import datetime


class MultiChannelConfig(BaseModel):
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


class MultiChannel:
    def __init__(self, config: MultiChannelConfig):
        self.config = config
        self.channels = [Channel(config) for config in self.config.channels]
        self.n = self.channels[0].n

    @property
    def x(self):
        x = {}
        for c in self.channels:
            x.update(c.x)
        return x

    @property
    def y(self):
        return {'y_dec': self.channels[0].y["left_y_dec"],
                'y_rate': self.channels[0].y["left_y_rate"]}

    @property
    def x_train(self):
        x = {}
        for c in self.channels:
            x.update(c.x_train.items())
        return x

    @property
    def x_test(self):
        x = {}
        for c in self.channels:
            x.update(c.x_test.items())
        return x

    @property
    def y_train(self):
        return {'y_dec': self.channels[0].y_train["left_y_dec"],
                'y_rate': self.channels[0].y_train["left_y_rate"]}

    @property
    def y_test(self):
        return {'y_dec': self.channels[0].y_test["left_y_dec"],
                'y_rate': self.channels[0].y_test["left_y_rate"]}

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
