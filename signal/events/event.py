import numpy as np

from signal.digital.digital_siginal import DigitalSignal
from signal.sequences.sequence import CompoundEvent


class Event(DigitalSignal):
    def __repr__(self):
        return f"Event(fs={self.fs}, duration={self.duration}, seed={self.seed})"

    def _generate_f(self) -> np.ndarray:
        """Default events is constant 1s * mag"""
        return np.ones(shape=(self.duration_pts,)) * self.mag
