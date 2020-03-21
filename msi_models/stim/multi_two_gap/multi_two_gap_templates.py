from datetime import datetime
from typing import List

import numpy as np

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events, template_sine_events


def template_unmatched(n_events: List[int] = None,
                       duration_tol: float = 0.5,
                       **kwargs) -> MultiTwoGapParams:
    if n_events is None:
        n_events = [11, 12, 13, 14, 15, 16]
    n_evs = np.random.choice(n_events, size=2, replace=False)
    seed = int(datetime.now().timestamp())

    config_1 = template_noisy_sine_events(n_events=n_evs[0],
                                          seed=seed,
                                          duration_tol=duration_tol,
                                          **kwargs)
    config_2 = template_sine_events(n_events=n_evs[1],
                                    seed=seed + 1,
                                    duration_tol=duration_tol,
                                    **kwargs)

    multi_params = MultiTwoGapParams(validate_as_matched=False,
                                     validate_as_sync=False,
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
                                     duration_tol=[config_1.duration_tol, config_2.duration_tol])

    return multi_params


def template_matched(n_events: List[int] = None,
                     duration_tol: float = 0.5,
                     **kwargs) -> MultiTwoGapParams:
    if n_events is None:
        n_events = [11, 12, 13, 14, 15, 16]
    n_ev = np.random.choice(n_events)
    seed = int(datetime.now().timestamp())

    config_1 = template_noisy_sine_events(n_events=n_ev,
                                          seed=seed,
                                          duration_tol=duration_tol,
                                          **kwargs)
    config_2 = template_sine_events(n_events=n_ev,
                                    seed=seed + 1,
                                    duration_tol=duration_tol,
                                    **kwargs)

    return MultiTwoGapParams(validate_as_matched=True,
                             validate_as_sync=False,
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
                             duration_tol=[config_1.duration_tol, config_2.duration_tol])


def template_sync(n_events: List[int] = None,
                  duration_tol: float = 0.5,
                  **kwargs) -> MultiTwoGapParams:
    if n_events is None:
        n_events = [11, 12, 13, 14, 15, 16]

    n_ev = np.random.choice(n_events)

    seed = int(datetime.now().timestamp())

    config = template_noisy_sine_events(n_events=n_ev, seed=seed,
                                        duration_tol=duration_tol,
                                        **kwargs)

    return MultiTwoGapParams(validate_as_matched=True,
                             validate_as_sync=True,
                             n_channels=2,
                             event=config.event,
                             n_events=config.n_events,
                             duration=config.duration,
                             background=config.background,
                             gap_1=config.gap_1,
                             gap_2=config.gap_2,
                             background_weight=config.background_weight,
                             seed=config.seed,
                             cache=config.cache,
                             duration_tol=config.duration_tol)
