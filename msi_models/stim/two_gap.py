from functools import partial
from itertools import zip_longest
from typing import Callable, Union, Tuple

import numpy as np
from audiodag.signal.components.component import Component, CompoundComponent
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent
from audiodag.signal.envelopes.templates import CosRiseEnvelope
from pydantic import BaseModel, PositiveInt, validator, root_validator

from msi_models.exceptions.params import IncompatibleParametersException, InvalidParameterException
from msi_models.stim.seeded import Seeded


def _callable_and_returns_expected_type(v, values, **kwargs):
    """This should run fast with lazy generation in audiodag components"""
    try:
        v_ = v()
    except Exception:
        raise InvalidParameterException(f"Invalid parameter")

    if not isinstance(v_, Component):
        raise IncompatibleParametersException(f"Events and gaps partial functions should return Components "
                                              f"(or CompoundComponents) not {type(v_)}")
    return v


class TwoGapParams(BaseModel):
    """
    Params object for TwoGapStim.

    :param duration: Total duration of stim.
    :param event: Partial function describing event. TODO: Can be swapped for an audiodag Component / CompoundComponent
                                                           when start time updates are supported
                                                           (see audiodag CompoundComponent._adjust_start)
    :param n_events: Number of events to add to stim.
    :param background: Background component, will be duration long.
    :param gap_1: Partial function describing gap_1. Again could be actual Component in future, and start time is
                  adjusted when sequence is generated.
    :param gap_2: Partial function describing gap_2
    :param seed: Int to generate numpy seed.
    :param cache: Bool to turn caching of generated stim on or off
    :param duration_tol: Acceptable deviation of total duration of selected gap combination, as proportion of duration.
    """
    duration: PositiveInt
    event: Union[partial, Callable]
    n_events: PositiveInt
    background: Union[partial, Callable]
    gap_1: Union[partial, Callable]
    gap_2: Union[partial, Callable]
    background_weight: float = 0
    seed: Union[PositiveInt, None] = None
    cache: bool = True
    duration_tol: float = 0.3

    class Config:
        """Allow setting of partial type."""
        arbitrary_types_allowed = True

    _validate_event = validator("event", allow_reuse=True)(_callable_and_returns_expected_type)
    _validate_background = validator("background", allow_reuse=True)(_callable_and_returns_expected_type)
    _validate_gap_1 = validator("gap_1", allow_reuse=True)(_callable_and_returns_expected_type)
    _validate_gap_2 = validator("gap_2", allow_reuse=True)(_callable_and_returns_expected_type)

    @root_validator
    def sampling_freqs_match(cls, values):
        """This should run fast with lazy generation in audiodag components"""
        fss = {k: values[k]().fs for k in ["gap_1", "gap_2", "event", "background"]}
        if len(np.unique(list(fss.values()))) > 1:
            raise IncompatibleParametersException(f"Sampling frequency mismatch between:"
                                                  f"({fss}")
        return values

    @classmethod
    def example_basic(cls,
                      duration: PositiveInt = 1000,
                      n_events: PositiveInt = 10,
                      fs: PositiveInt = 1000,
                      gap_1_duration: PositiveInt = 25,
                      gap_2_duration: PositiveInt = 50) -> "TwoGapParams":
        event_sine_component = SineComponent(duration=gap_1_duration, freq=8, fs=fs, mag=2)
        event_noise_component = NoiseComponent(duration=gap_1_duration, fs=fs, mag=0.02)
        event = partial(CompoundComponent,
                        events=[event_sine_component, event_noise_component])

        gap_1 = partial(NoiseComponent, duration=gap_1_duration, mag=0.02, fs=fs)
        gap_2 = partial(NoiseComponent, duration=gap_2_duration, mag=0.02, fs=fs)
        background = partial(NoiseComponent, duration=duration, fs=fs, mag=0.02,
                             envelope=partial(CosRiseEnvelope, rise=20))

        return TwoGapParams(duration=duration, n_events=n_events, event=event, gap_1=gap_1, gap_2=gap_2,
                            background=background, background_weight=2)


class TwoGapStim(Seeded):
    def __init__(self, params: TwoGapParams):
        super().__init__(seed=params.seed)
        self.params = params

        event = self.params.event()
        self._indicator_event = partial(Component, duration=event.duration, fs=event.fs, mag=1)
        self._indicator_background = partial(Component, duration=params.duration, fs=event.fs, mag=0)
        self._y = None
        self._y_true = None

        self.pool_ = None
        self.ev_list_ = None
        self.gap_ns_ = None
        self.active_duration_ = None

    @classmethod
    def construct_from_basic(cls, **kwargs) -> "TwoGapStim":
        return TwoGapStim(TwoGapParams.example_basic(**kwargs))

    def _generate(self) -> CompoundComponent:
        self.state = self.seed
        self._generate_valid_combinations()
        self._generate_sequence()

        return self._generate_events()

    def _generate_valid_combinations(self):
        """
        Generate valid event and gap combinations.

        Validity depends on total duration (+/- tolerance), gap durations, number of events.

        Generates
            self.gap_ns: List of valid possible combinations of [gap_1, gap_2]
        """

        # Calculate time committed to event
        event_time = self.params.n_events * self.params.gap_1.keywords['duration']

        # Make grid of possible n gaps (axis 0 = gap_1, axis 1 = gap_2)
        # Min number of either events is 0. Max is n_events + 1. Must add up to n_events + 1, ie. the diags of:
        gaps = np.arange(0, self.params.n_events + 2, 1)
        g1s, g2s = np.meshgrid(gaps, gaps)
        diag_mask = np.fliplr(np.eye(len(gaps), len(gaps),
                                     dtype=bool))

        # Calculate total time for each element
        t_g1s = g1s * self.params.gap_1().duration
        t_g2s = g2s * self.params.gap_2().duration
        t_total = np.array(t_g1s + t_g2s + event_time)

        # Get the indexes in grid where timing works for gap combo
        # ie. Duration here inside requested duration +/- tolerance.
        gap_idxs = np.argwhere((np.abs(t_total - self.params.duration) < (t_total * self.params.duration_tol))
                               & diag_mask)

        if len(gap_idxs) == 0:
            raise IncompatibleParametersException(f"No valid durations within duration tolerance "
                                                  f"({self.params.duration_tol})")

        # Get the actual event numbers
        self.gap_ns_ = [[gaps[gs[1]], gaps[gs[0]]] for gs in gap_idxs]

    def _generate_sequence(self):
        """
        Generate shuffled sequence of events (as partial functions).
        """

        # Pick a valid combination of events
        self.pool_ = self.gap_ns_[np.random.randint(0, len(self.gap_ns_))]
        # Create the gaps and shuffle order
        gap_list = [self.params.gap_1] * self.pool_[0] + [self.params.gap_2] * self.pool_[1]
        np.random.shuffle(gap_list)

        # Create the events and zip with gaps so [gap_x, event, gap_x, event ... gap_x]
        event_list = [self.params.event] * self.params.n_events
        self.ev_list_ = [ge for pair in list(zip_longest(gap_list, event_list)) for ge in pair if ge is not None]

        # Given the combinations of events chosen, record to "active" duration
        self.active_duration_ = (self.pool_[0] * self.params.gap_1().duration
                                 + self.pool_[1] * self.params.gap_2().duration
                                 + self.params.n_events * self.params.event().duration)

    def _generate_events(self) ->Tuple[CompoundComponent, CompoundComponent]:
        """
        Generate a CompoundComponent from the prepared event sequence.
        """

        # Pick a start cursor so the active duration is randomly placed entirely inside total duration
        cursor = np.random.randint(self.params.duration - self.active_duration_)

        # Generate actual events with now-known start times. Also make make indicating event locations.
        evs = [self.params.background()]
        weights = [self.params.background_weight]
        indicators = [self._indicator_background()]
        for ev in self.ev_list_:
            ev_init = ev(start=cursor)

            evs.append(ev_init)
            weights.append(1)

            if ev == self.params.event:
                indicators.append(self._indicator_event(start=cursor))

            cursor += ev_init.duration

        return (CompoundComponent(events=evs, weights=weights),
                CompoundComponent(events=indicators))

    def _get_or_generate(self) -> Tuple[CompoundComponent]:

        if (self._y is None) or (self._y_true is None):
            y, y_true = self._generate()
        else:
            y = self._y
            y_true = self._y_true

        if self.params.cache:
            self._y = y
            self._y_true = y_true

        return y, y_true

    @property
    def y(self) -> CompoundComponent:
        return self._get_or_generate()[0]

    @property
    def y_true(self) -> CompoundComponent:
        return self._get_or_generate()[1]


if __name__ == "__main__":

    # Example stim:
    stim = TwoGapStim.construct_from_basic()
    stim.y.plot(show=False)
    stim.y_true.plot(show=True)

    # Example stims with increasing events
    for n in range(8, 14):
        try:
            stim = TwoGapStim.construct_from_basic(n_events=n)
            stim.y.plot(show=False)
            stim.y_true.plot(show=True)

        except IncompatibleParametersException as e:
            print(e)
