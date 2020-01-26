from functools import partial
from typing import Union, List, Tuple

import numpy as np
from audiodag.signal.events.event import CompoundEvent
from audiodag.signal.events.noise import NoiseEvent
from audiodag.signal.events.tonal import SineEvent


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
        self._y: Union[None, CompoundEvent] = None

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


class TwoGap(Seeded):
    def __init__(self,
                 duration: int,
                 event: partial,
                 n_events: int,
                 gap_1: partial,
                 gap_2: partial,
                 cache: bool = True,
                 duration_tol: float = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.duration = duration
        self.event = event
        self.n_events = n_events
        self.gap_1 = gap_1
        self.gap_2 = gap_2
        self.cache = cache
        self.duration_tol = duration_tol

    @classmethod
    def construct_from_basic(cls,
                             n_events: int = 10,
                             duration: int = 900,
                             event_mag: float = 2,
                             event_noise_mag: float = 0.02,
                             gap_1_duration: int = 25,
                             gap_2_duration: int = 50,
                             noise_mag: float = 0.02,
                             fs: int = 1000) -> "TwoGap":

        event_sine = SineEvent(duration=gap_1_duration,
                               freq=8,
                               fs=fs,
                               mag=event_mag)
        event_noise = NoiseEvent(duration=gap_1_duration,
                                 fs=fs,
                                 mag=event_noise_mag)

        return TwoGap(n_events=n_events,
                      duration=duration,
                      event=partial(CompoundEvent,
                                    events=[event_sine, event_noise]),
                      gap_1=partial(NoiseEvent,
                                    duration=gap_1_duration,
                                    mag=noise_mag,
                                    fs=fs),
                      gap_2=partial(NoiseEvent,
                                    duration=gap_2_duration,
                                    mag=noise_mag,
                                    fs=fs))

    def _generate(self) -> CompoundEvent:
        self.state = self.seed
        self._generate_valid_combinations()
        self._generate_sequence()
        return self._generate_compound_event()

    def _generate_valid_combinations(self):
        """
        Generate valid event and gap combinations.

        Validity depends on total duration (+/- tolerance), gap durations, number of events.

        Generates
            self.gap_ns: List of valid possible combinations of [gap_1, gap_2]
        """

        # Calculate time committed to event
        event_time = self.n_events * self.gap_1.keywords['duration']

        # Make grid of possible n gaps (axis 0 = gap_1, axis 1 = gap_2)
        gaps = np.arange(int(self.n_events / 2), self.n_events + 1, 1)
        g1s, g2s = np.meshgrid(gaps, gaps)

        # Calculate total time for each element
        t_g1s = g1s * self.gap_1.keywords['duration']
        t_g2s = g2s * self.gap_2.keywords['duration']
        t_total = np.array(t_g1s + t_g2s + event_time)

        # Get the indexes in grid where timing works for gap combo
        # ie. Duration here inside requested duration +/- tolerance.
        gap_idxs = np.argwhere(np.abs((t_total - self.duration)) < t_total * self.duration_tol)

        # Get the actual event numbers
        self.gap_ns_ = [[gaps[gs[0]], gaps[gs[1]]] for gs in gap_idxs]

    def _generate_sequence(self):
        """
        Generate sequence of events, as partial functions.
        """

        # Pick a random pool
        self.pool_ = self.gap_ns_[np.random.randint(0, len(self.gap_ns_))]

        # Select start event
        g_n, pool = self.select_gap(self.pool_)
        ev_list = [self.gap_1 if g_n == 0 else self.gap_2]
        for ev in range(0, self.n_events):
            g_n, pool = self.select_gap(pool)

            # Update event list
            ev_list += [self.event, self.gap_1 if g_n == 0 else self.gap_2]

        self.ev_list_ = ev_list

    def _generate_compound_event(self) -> CompoundEvent:
        """
        Generate a CompoundEvent from the prepared event sequence.
        """
        cursor = 0
        evs = []
        for ev in self.ev_list_:
            # Run partial to generate event
            ev_init = ev(start=cursor)
            cursor += ev_init.duration
            evs.append(ev_init)

        return CompoundEvent(events=evs)

    @staticmethod
    def select_gap(pool: List[int]) -> Tuple[int, List[int]]:
        if np.all([p > 0 for p in pool]):
            # Pick gap 1 or 2
            g_n = np.random.randint(0, 2)
        elif pool[0] == 0:
            g_n = 1
        elif pool[1] == 0:
            g_n = 0

        pool[g_n] -= 1

        return g_n, pool

    @property
    def y(self) -> CompoundEvent:

        if self._y is None:
            y = self._generate()
        else:
            y = self._y

        if self.cache:
            self._y = y

        return y


if __name__ == "__main__":
    stim = TwoGap.construct_from_basic()
    stim.y.plot_subplots(show=True)
