import os
import pathlib
from typing import Union, List, Dict

import numpy as np

from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel
from msi_models.stimset.multi_channel_config import MultiChannelConfig


class ExperimentalDataset:
    def __init__(self, name: str = 'unnamed_dataset', n: int = 1000, difficulty: int = 50):
        """
        :param difficulty: Difficulty of data generated, (easy) 0 -> 100 (hard).
        :param n: Number of instances to generate.
        """
        self.name = name
        self.n = n
        self.difficulty = difficulty

        # Parameters used to generate data, fixed for now
        self.generate_params = {"templates": [MultiTwoGapTemplate['left_only'], MultiTwoGapTemplate['right_only'],
                                              MultiTwoGapTemplate['matched_sync'], MultiTwoGapTemplate['matched_async'],
                                              MultiTwoGapTemplate['unmatched_async']],
                                "fs": 500}
        # Parameters used to control the template used in generation
        self.template_kwargs = {"duration": 1300, "duration_tol": 0.5, 'background_mag': difficulty / 2 / 100}

        # Parameters used to prepare data reader
        self.common_channel_kwargs: Dict[str, Union[float, str, List[str]]] = {"train_prop": 0.8,
                                                                               "x_keys": ["x", "x_mask"],
                                                                               "y_keys": ["y_rate", "y_dec"]}

        # Set on .build()
        self.path: Union[None, str] = None
        self.rates: List[int] = []
        self.types: List[int] = []
        self._mc: Union[None, MultiChannel] = None

    def __hash__(self) -> int:
        return hash(self.path)

    def build(self, path: str = "sample_multisensory_data_mix_med_1k.hdf5", n_jobs: int = 12,
              batch_size: int = None) -> "ExperimentalDataset":
        """
        :param path: Path to load from (and/or to create at).
        :param n_jobs: Number of jobs to use if generating data.
        :param batch_size: Size of batches to use in generation. Default None, which sets automatically (may result in
                           rounding errors on n when n_jobs != 1).
        """

        self.path = path.replace('\\', '/')
        self._generate(n_jobs, batch_size)
        self._load(self.path)

        return self

    def _generate(self, n_jobs: int, batch_size: int = None) -> None:
        if batch_size is None:
            if n_jobs == 1:
                batch_size = self.n
            else:
                batch_size = max(1, int(np.log2(self.n / n_jobs)))

        if not os.path.exists(self.path):
            pathlib.Path(os.path.split(self.path)[0]).mkdir(exist_ok=True)
            MultiTwoGapStim.generate(fn=self.path, n=self.n, n_jobs=n_jobs,
                                     batch_size=batch_size,
                                     template_kwargs=self.template_kwargs, **self.generate_params)

    def _load(self, path: str):
        self.common_channel_kwargs.update({'path': path})

        multi_config = MultiChannelConfig(path=path, key='agg', y_keys=self.common_channel_kwargs["y_keys"],
                                          channels=[ChannelConfig(key='left', **self.common_channel_kwargs),
                                                    ChannelConfig(key='right', **self.common_channel_kwargs)])
        self._mc = MultiChannel(multi_config)

        self.types = list(np.sort(self._mc.summary.type.unique()))
        rates = np.unique(
            list(self._mc.summary.left_n_events.unique()) + list(self._mc.summary.right_n_events.unique()))
        self.rates = [r for r in rates if r != 0]

    @property
    def mc(self) -> MultiChannel:
        if self._mc is None:
            raise FileNotFoundError(f"Data not ready.")

        return self._mc


if __name__ == "__main__":
    exp_data = ExperimentalDataset()
    exp_data.build('test.hdf')
