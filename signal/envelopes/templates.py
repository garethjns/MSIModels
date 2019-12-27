import numpy as np

from signal.digital.conversion import ms_to_pts
from signal.envelopes.envelope import Envelope


class ConstantEnvelope(Envelope):
    def f(self, y: np.ndarray) -> np.ndarray:
        return y


class CosEnvelope(Envelope):
    def f(self, y: np.ndarray) -> np.ndarray:
        return y * (np.cos(np.linspace(1 * np.pi, 3 * np.pi, len(y))) + 1) * 0.5


class CosRiseEnvelope(Envelope):
    rise: int

    @property
    def rise_pts(self) -> int:
        return ms_to_pts(fs=self.fs,
                         t_ms=self.rise)

    def f(self, y: np.ndarray) -> np.ndarray:
        cos_rise = (np.cos(np.linspace(1 * np.pi, 2 * np.pi, self.rise_pts)) + 1) * 0.5
        cos_fall = np.flip(cos_rise)

        uniform_centre = np.ones(shape=(len(y) - 2 * self.rise_pts))

        envelope = np.concatenate((cos_rise, uniform_centre, cos_fall),
                                  axis=0)

        return y * envelope
