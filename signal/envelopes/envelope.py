from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Envelope(ABC):
    def __init__(self,
                 fs: Union[int, None] = None,
                 **kwargs):
        self.fs = fs

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.f(y)

    def __str__(self):
        return f"{self.__name__}"

    @abstractmethod
    def f(self, y: np.ndarray) -> np.ndarray:
        pass
