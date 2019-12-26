from functools import partial, reduce
from typing import List, Callable

import numpy as np

from stim.base_signal import DigitalSignal


class Event(DigitalSignal):
    def __repr__(self):
        return f"Event(fs={self.fs}, duration={self.duration}, seed={self.seed})"

    def _generate_f(self) -> np.ndarray:
        """Default event is constant 1s * mag"""
        return np.ones(shape=(self.duration_pts,)) * self.mag

    def __mul__(self, other):
        """
        Multiplying events generates a CompoundEvent object with a generator for the combined signals.

        Weighting is even.
        """
        return CompoundEvent(events=[self, other])


class NoiseEvent(Event):
    """Class specifically for noisy events."""
    def __init__(self,
                 dist: str = 'normal',
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.dist = dist

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist: str) -> None:
        if dist.lower() not in ['normal', 'uniform']:
            raise ValueError(f"Dist {dist} is not valid.")

        self._dist = dist

    def _generate_f(self) -> np.ndarray:
        """Sample noise from the RandomState."""
        return getattr(self.state, self.dist)(size=(self.duration_pts,)) * self.mag


class SineEvent(Event):
    """Class specifically for tonal events."""
    def __init__(self,
                 freq: int = 2000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.freq = freq

    def _generate_f(self) -> np.ndarray:
        """Generate vector for event"""
        return np.sin(np.linspace(0, 4 * np.pi * self.freq, self.duration_pts)) * 0.5 * self.mag


class CompoundEvent(Event):
    """
    Object for combining events, for example adding noise to another event.

    Supports combination of multiple events, but only with equal weighting and same durations for now.
    """
    def __init__(self, events: List[Event]):

        self._verify_event_list(events)

        super().__init__(fs=events[0].fs,
                         duration=events[0].duration)

        self.weights = [1 / len(events) for _ in range(len(events))]
        self.events = events

        self._generate_f = self._make_generate_f()

    @staticmethod
    def _verify_event_list(events: List[Event]):
        check_params = ['duration', 'fs']

        for p in check_params:
            param_values = [getattr(e, p) for e in events]
            if len(np.unique(param_values)) > 1:
                raise ValueError(f"Param {p} is inconsistent across events: {param_values}")

    def _envelope_f(self) -> np.ndarray:
        """
        Envelope function.

        The Envelope is applied at the individual event levels before combination. Another envelope can be here, or not.
        This implementation overloads the cosine envelope with a uniform envelope.
        """
        return np.ones(shape=(self.duration_pts,))

    @staticmethod
    def _combiner(ev_1: Event, ev_2: Event,
                  weight: float = 0.5) -> np.ndarray:
        start = int(min(min(ev_1.x_pts), min(ev_2.x_pts)))
        end = int(max(max(ev_1.x_pts), max(ev_2.x_pts)))
        buffered_len = end - start + 1

        x = np.linspace(start, end, buffered_len,
                        dtype=int)
        y = np.zeros(shape=(buffered_len,))

        y[ev_1.x_pts] = y[ev_1.x_pts] + ev_1.y * weight
        y[ev_2.x_pts] = y[ev_2.x_pts] + ev_2.y * (1 - weight)

        return x, y

    def _make_generate_f(self) -> Callable:
        """
        Make the generator function.

        Returns as Callable that does the combination using the .y properties on each Event. This callable can be
        assigned to ._generate_f which maintains the Event API.
        """
        return partial(reduce, lambda x, y: x + y * 1 / len(self.events), [e.y for e in self.events])