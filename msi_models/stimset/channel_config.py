from typing import List

import h5py
import numpy as np
from pydantic import BaseModel, root_validator
from pydantic.types import FilePath

from msi_models.exceptions.params import InvalidParameterException


class ChannelConfig(BaseModel):
    """
    Config for an individual channel containing the x data and y data.

    :param path: Path to hdf5 file containing x_keys and y_keys.
    :param y_keys: The keys for x inputs (currently supported: x, x_mask)
    :param y_keys: The keys for the y_rate and y_dec.
    :param key: Root key to use to get the other keys. If file contains one channel, this can be empty '' or '/'. If the
                file contains more than a single channel, use this key to specific which to laod, eg 'left/' or
                'right/'.
    :param seed: Int specifying a numpy seed. Used for the train/test split
    :param train_prop: Train proportion for train/test split. Remaining data will be kept for testing.
    """
    path: FilePath
    x_keys: List[str] = []
    y_keys: List[str] = []
    key: str = ''
    seed: int = 0
    train_prop: float = 0.8

    @root_validator
    def all_keys_exist_and_match_len(cls, values):
        with h5py.File(values['path'], 'r') as f:

            key = "/" if values["key"] == "" else values["key"]
            keys = list(f[key].keys())

        if not np.all([v in keys for v in values["x_keys"]]):
            raise InvalidParameterException(f"Some of x_keys ({values['x_keys']}) missing from file keys ({keys})")

        if not np.all([v in keys for v in values["y_keys"]]):
            raise InvalidParameterException(f"Some of y_keys ({values['y_keys']}) missing from file keys ({keys})")

        return values
