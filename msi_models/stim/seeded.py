import numpy as np

from typing import Union
from audiodag.signal.components.component import CompoundComponent


class Seeded:

    def __init__(self,
                 seed: Union[int, None] = None) -> None:
        """
        :param seed: Integer used to set numpy RandomState.
        """
        self.seed = seed
        self.state = seed

        self._seed: int
        self._state: np.random.RandomState
        self._y: Union[None, CompoundComponent] = None

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