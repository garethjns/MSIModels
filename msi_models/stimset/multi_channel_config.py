from datetime import datetime
from typing import List, Union

import h5py
import numpy as np
from pydantic import BaseModel, validator, root_validator
from pydantic.types import FilePath, PositiveInt

from msi_models.exceptions.params import IncompatibleParametersException, InvalidParameterException
from msi_models.stimset.channel_config import ChannelConfig


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
