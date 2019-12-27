from abc import ABC, abstractmethod

import numpy as np


class Envelope(ABC):
    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.f(y)

    @abstractmethod
    def f(self, y: np.ndarray) -> np.ndarray:
        pass
