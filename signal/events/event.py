from functools import reduce, partial
from typing import List, Tuple, Callable

import numpy as np

from signal.digital.conversion import pts_to_ms
from signal.digital.digital_siginal import DigitalSignal
from signal.envelopes.templates import ConstantEnvelope


class Event(DigitalSignal):
    def __repr__(self):
        return f"Event(fs={self.fs}, duration={self.duration}, seed={self.seed})"

    def _generate_f(self) -> np.ndarray:
        """Default events is constant 1s * mag"""
        return np.ones(shape=(self.duration_pts,)) * self.mag

    def __mul__(self, other):
        """
        Multiplying events generates a CompoundEvent object with a generator for the combined signals.

        Weighting is even.
        """

        return CompoundEvent(events=[self, other])


class CompoundEvent(Event):
    """
    Object for combining events, for example adding noise to another events.

    Supports combination of multiple events, but only with equal weighting and same durations for now.
    """
    def __init__(self, events: List[Event],
                 weights: List[float] = None,
                 envelope = ConstantEnvelope):

        self._verify_event_list(events)
        start, _, duration = self._new_duration(events)

        super().__init__(fs=events[0].fs,
                         start=start,
                         duration=pts_to_ms(duration,
                                            fs=events[0].fs))

        if weights is None:
            weights = [1 / len(events) for _ in range(len(events))]
        self.weights = weights
        self.events = events

        self._generate_f = self._make_generate_f()

    @staticmethod
    def _verify_event_list(events: List[Event]):
        check_params = ['fs']

        for p in check_params:
            param_values = [getattr(e, p) for e in events]
            if len(np.unique(param_values)) > 1:
                raise ValueError(f"Param {p} is inconsistent across events: {param_values}")

    @staticmethod
    def _new_duration(events: List[Event]):
        start = reduce(lambda ev_a, ev_b: min(ev_a, ev_b), [e.x_pts.min() for e in events])
        end = reduce(lambda ev_a, ev_b: max(ev_a, ev_b), [e.x_pts.max() for e in events])
        return start, end, end - start + 1

    def _combiner(self,
                  events: List[Event],
                  weights: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:

        if weights is None:
            weights = [1 / len(events) for _ in events]

        y = np.zeros(shape=(len(events), self.duration_pts))
        for e_i, (e, w) in enumerate(zip(events, weights)):
            y[e_i, e.x_pts - self.start] = e.y * w

        y = y.sum(axis=0)

        return y

    def _make_generate_f(self) -> Callable:
        """
        Make the generator function.

        Returns as Callable that does the combination using the .y properties on each Event. This callable can be
        assigned to ._generate_f which maintains the Event API.
        """
        return partial(lambda: self._combiner(events=self.events,
                                              weights=self.weights))