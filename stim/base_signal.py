import gc
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import matplotlib.pyplot as plt


def ms_to_pts(t_ms: int, fs: int) -> int:
    """Convert time in ms to time in pts."""
    return int(fs * t_ms / 1000)


def pts_to_ms(t_pts: int, fs: int) -> float:
    """Convert time in ms to time in pts."""
    return t_pts * 1000 / fs


class DigitalSignal(ABC):
    """
    Object representing basic properties of a digital signal, handles time <-> samples conversion. Doesn't bother
    with with analog <-> bits scale on y yet, though.
    """
    def __init__(self,
                 start: int = 0,
                 fs: int = 1000,
                 duration: int = 20,
                 mag: int = 1,
                 clip: float = 2,
                 seed: Union[int, None] = None,
                 cache: bool = False) -> None:
        """

        :param start: Start time, in ms.
        :param fs: Sampling rate.
        :param duration: Duration in ms.
        :param mag: Magnitude of signal in volts.
        :param clip: Max positive magnitude of signal.
        :param seed: Integer used to set numpy RandomState used for generating stochastic signals.
        :param cache: If True, hold signal in memory after generation. Otherwise generate each time it's accessed.
        """
        self.start = start
        self.duration = duration
        self.mag = mag
        self.fs = fs
        self.seed = seed
        self.state = seed
        self.cache = cache
        self.clip = clip

        self._y: Union[np.ndarray, None] = None
        self._seed: int
        self._state: np.random.RandomState

    @abstractmethod
    def __repr__(self):
        """__repr__ is used for id and eq, it should be redefined in children."""
        return f"DigitalTime(fs={self.fs}, duration={self.duration}, seed={self.state})"

    def __hash__(self):
        """Hash assumes important parameters are included in __repr__."""
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        """Ordering is undefined, but this is necessary for things like np.unique()."""
        return self.__hash__() < other.__hash__()

    def clear(self):
        """Remove the signal vector from memory."""
        self._y = None
        gc.collect()

    @abstractmethod
    def _generate_f(self) -> np.ndarray:
        """Function to generate Signal. This should be overloaded in child."""
        return np.zeros(shape=(self.duration_pts,))

    def _envelope_f(self) -> np.ndarray:
        """Envelope function for whole signal - default is 180deg of a cosine."""
        return (np.cos(np.linspace(1 * np.pi, 3 * np.pi, self.duration_pts)) + 1) * 0.5

    def _generate(self) -> np.ndarray:
        """
        Generate the signal.

        Combines the generator function, the envelope, and applies clipping.
        """
        y = self._generate_f() * self._envelope_f()
        y[y > self.clip] = self.clip

        return y

    @property
    def x(self) -> np.ndarray:
        """Time axis, in ms."""
        return np.linspace(self.start, self.start + self.duration, self.duration_pts)

    @property
    def x_pts(self) -> np.ndarray:
        """Time axis, in samples."""
        return np.linspace(self.start_pts, self.start_pts + self.duration_pts - 1, self.duration_pts,
                           dtype=int)

    @property
    def y(self) -> np.ndarray:
        """Signal. Generated on the fly if required."""
        if self._y is None:
            self.state = self.seed
            y = self._generate()
        else:
            y = self._y

        if self.cache:
            self._y = y

        return y

    def plot(self,
             show: bool = False):
        """Plot the signal against time in ms."""
        plt.plot(self.x, self.y)

        if show:
            plt.show()

    @property
    def duration_pts(self) -> int:
        """Duration of the signal, in ms."""
        return ms_to_pts(t_ms=self.duration,
                         fs=self.fs)

    @property
    def start_pts(self) -> float:
        """Duration of the signal, in samples."""
        return ms_to_pts(t_ms=self.start,
                         fs=self.fs)

    @property
    def seed(self) -> int:
        """Return the seed used to generate the signal."""
        return self._seed

    @seed.setter
    def seed(self, seed: Union[int, None]) -> None:
        """Set the seed by generating a RandomState from the input."""

        if seed is None:
            seed = np.random.RandomState(seed=seed).randint(2 ** 31)

        self.state = seed
        self._seed = seed

    @property
    def state(self) -> np.random.RandomState:
        return self._state
    
    @state.setter
    def state(self, seed) -> None:
        self._state = np.random.RandomState(seed)
