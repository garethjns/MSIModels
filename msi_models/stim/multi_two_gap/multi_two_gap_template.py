from datetime import datetime
from enum import Enum
from typing import List, Union

import numpy as np

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.two_gap.two_gap_params import TwoGapParams
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events


class MultiTwoGapTemplate(Enum):
    """
    Defines stimulus 5 type templates.

    1) Left channels contains events, right is noise only (also "unmatched" and "independent")
    2) Right channel contains events, left is noise only (also "unmatched" and "independent")
    3) Both channels contain the same number of events ("matched"), and each event occurs synchronously between channels
       ("sync").
    4) Both channels contain the same number of events ("matched"), but these don't necessarily occur synchronously
       across channels ("async")
    5) Both channels indicate the same decision, but the number of events can differ ("unmatched") and cannot be
       synchronous ("async").
    6) Channels don't necessarily indicate the same decision ("independent"), don't necessarily have the same number of
       events ("unmatched"), and cannot be synchronous ("async").

    Some parameters are flexible, these can be set on __init__.
     - n_events
     - duration_tol

    """
    left_only = 1
    right_only = 2
    matched_sync = 3
    matched_async = 4
    unmatched_async = 5
    unmatched_async_independent = 6

    n_events: List[int]
    duration: int
    background_mag: int
    duration_tol: float
    fs: int
    cache: True
    _seed: Union[None, int]

    _template: Union[None, MultiTwoGapParams]

    def set_options(self, fs: int = 1000, n_events: List[int] = None, duration_tol: float = 0.5, duration: int = 1300,
                    background_mag: int = 0.1, cache: bool = True, seed: int = None):
        if n_events is None:
            n_events = [11, 12, 13, 14, 15, 16]
        self.n_events = n_events
        self.duration_tol = duration_tol
        self.fs = fs
        self.duration = duration
        self.background_mag = background_mag
        self.cache = cache
        self._seed = seed

        return self

    @property
    def seed(self):
        return self._seed if self._seed is not None else int(datetime.now().timestamp())

    def build(self) -> MultiTwoGapParams:

        self._template = None

        if self == MultiTwoGapTemplate.left_only:
            self._template = self._build_template_left_only()

        if self == MultiTwoGapTemplate.right_only:
            self._template = self._build_template_right_only()

        if self == MultiTwoGapTemplate.matched_sync:
            self._template = self._build_template_matched_sync()

        if self == MultiTwoGapTemplate.matched_async:
            self._template = self._build_template_matched_async()

        if self == MultiTwoGapTemplate.unmatched_async:
            self._template = self._build_template_unmatched_async()

        if self == MultiTwoGapTemplate.unmatched_async_independent:
            raise NotImplementedError

        return self._template

    def example(self) -> None:
        MultiTwoGapStim(self.set_options().build()).plot(show=True)

    @staticmethod
    def _multi_config_from_configs(config_1: TwoGapParams, config_2: TwoGapParams, validate_as_matched=False,
                                   validate_as_sync=False) -> MultiTwoGapParams:

        return MultiTwoGapParams(validate_as_matched=validate_as_matched, validate_as_sync=validate_as_sync,
                                 n_channels=2,
                                 event=[config_1.event, config_2.event],
                                 n_events=[config_1.n_events, config_2.n_events],
                                 duration=[config_1.duration, config_2.duration],
                                 background=[config_1.background, config_2.background],
                                 gap_1=[config_1.gap_1, config_2.gap_1],
                                 gap_2=[config_1.gap_2, config_2.gap_2],
                                 background_weight=[config_1.background_weight, config_2.background_weight],
                                 seed=[config_1.seed, config_2.seed],
                                 cache=[config_1.cache, config_2.cache],
                                 duration_tol=[config_1.duration_tol, config_2.duration_tol],
                                 normalise=[config_1.normalise, config_2.normalise],
                                 normalise_across_channels=True)

    def _build_template_left_only(self) -> MultiTwoGapParams:
        n_evs = np.random.RandomState(self.seed).choice(self.n_events, size=1, replace=False)

        config_1 = template_noisy_sine_events(n_events=n_evs, seed=self.seed + 1, duration_tol=9999999, normalise=False,
                                              fs=self.fs, duration=self.duration, background_mag=self.background_mag,
                                              cache=self.cache)
        config_2 = template_noisy_sine_events(n_events=0, seed=self.seed + 1, duration_tol=9999999, normalise=False,
                                              fs=self.fs, duration=self.duration, background_mag=self.background_mag,
                                              cache=self.cache)

        return self._multi_config_from_configs(config_1, config_2, validate_as_sync=False, validate_as_matched=False)

    def _build_template_right_only(self) -> MultiTwoGapParams:
        n_evs = np.random.RandomState(self.seed).choice(self.n_events, size=1, replace=False)

        config_1 = template_noisy_sine_events(n_events=0, seed=self.seed + 1, duration_tol=9999999, normalise=False,
                                              fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)
        config_2 = template_noisy_sine_events(n_events=n_evs, seed=self.seed + 1, duration_tol=9999999, normalise=False,
                                              fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)

        return self._multi_config_from_configs(config_1, config_2, validate_as_sync=False, validate_as_matched=False)

    def _build_template_unmatched_async(self) -> MultiTwoGapParams:
        n_evs = np.random.RandomState(self.seed).choice(self.n_events, size=2, replace=False)

        config_1 = template_noisy_sine_events(n_events=n_evs[0], seed=self.seed, duration_tol=self.duration_tol,
                                              normalise=False, fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)
        config_2 = template_noisy_sine_events(n_events=n_evs[1], seed=self.seed + 1, duration_tol=self.duration_tol,
                                              normalise=False, fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)

        return self._multi_config_from_configs(config_1, config_2, validate_as_sync=False, validate_as_matched=False)

    def _build_template_matched_async(self) -> MultiTwoGapParams:
        n_evs = np.random.RandomState(self.seed).choice(self.n_events, size=1, replace=False)

        config_1 = template_noisy_sine_events(n_events=n_evs, seed=self.seed, duration_tol=self.duration_tol,
                                              normalise=False, fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)
        config_2 = template_noisy_sine_events(n_events=n_evs, seed=self.seed + 1, duration_tol=self.duration_tol,
                                              normalise=False, fs=self.fs, duration=self.duration, cache=self.cache,
                                              background_mag=self.background_mag)

        return self._multi_config_from_configs(config_1, config_2, validate_as_sync=False, validate_as_matched=True)

    def _build_template_matched_sync(self) -> MultiTwoGapParams:
        n_evs = np.random.RandomState(self.seed).choice(self.n_events, size=1, replace=False)

        config = template_noisy_sine_events(n_events=n_evs, seed=self.seed, duration_tol=self.duration_tol,
                                            normalise=False, fs=self.fs, duration=self.duration,
                                            background_mag=self.background_mag, cache=self.cache)

        return self._multi_config_from_configs(config, config, validate_as_sync=True, validate_as_matched=True)


if __name__ == "__main__":
    MultiTwoGapTemplate(1).example()
    MultiTwoGapTemplate(2).example()
    MultiTwoGapTemplate(3).example()
    MultiTwoGapTemplate(4).example()
    MultiTwoGapTemplate(5).example()
