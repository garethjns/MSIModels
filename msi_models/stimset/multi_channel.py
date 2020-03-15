from dataclasses import dataclass

from msi_models.stimset.channel import Channel
from typing import List


@dataclass
class MultiChannel:
    _channels: List[Channel]
