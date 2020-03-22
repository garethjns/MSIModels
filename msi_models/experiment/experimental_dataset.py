import copy
from dataclasses import dataclass
from typing import Union

from msi_models.stimset.channel import ChannelConfig, Channel
from msi_models.stimset.multi_channel import MultiChannel, MultiChannelConfig


@dataclass
class ExperimentalDataset:
    config: Union[ChannelConfig, MultiChannelConfig]
    name: str = "unnamed_dataset"
    _stimset: Union[MultiChannel, ChannelConfig] = None
    _seed: int = None

    def build(self, seed):
        self.config.seed = seed

        if isinstance(self.config, ChannelConfig):
            self._stimset = Channel(self.config)
        else:
            self._stimset = MultiChannel(self.config)

        self._seed = seed

    @property
    def stimset(self):
        if self._stimset is None:
            self.build(seed=self._seed)

        return self._stimset
