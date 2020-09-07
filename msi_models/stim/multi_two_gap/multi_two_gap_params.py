from datetime import datetime
from functools import partial
from typing import List, Union, Callable

import numpy as np
from pydantic import BaseModel, validator, root_validator

from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events


def _ensure_list(v, values, **kwargs):
    if not isinstance(v, list):
        v = [v] * values["n_channels"]
    return v


class MultiTwoGapParams(BaseModel):
    """
    Config for multichannel two gap stim, designed initially to be 2 channel.

    All params can either be single value (common between channels) or list in order [c1, c2].

    Normalise has special behaviour.
    If List[bool, bool] applies normalisation to each channel separately
    If True: Turns OFF individual stim normalisation, then normalises across channels after generation
    If False, off completely
    """
    # Multi event params
    n_channels: int = 2
    # Individual event params
    event: Union[Union[partial, Callable],
                 List[Union[partial, Callable]]]
    duration: Union[int, List[int]]
    n_events: Union[int, List[int]]
    background: Union[Union[partial, Callable],
                      List[Union[partial, Callable]]]
    gap_1: Union[Union[partial, Callable],
                 List[Union[partial, Callable]]]
    gap_2: Union[Union[partial, Callable],
                 List[Union[partial, Callable]]]
    background_weight: Union[float, List[float]]
    seed: Union[Union[int, None], List[Union[int, None]]]
    cache: Union[bool, List[bool]]
    duration_tol: Union[float, List[float]]
    normalise: Union[bool, List[bool]] = [False, False]
    normalise_across_channels: bool = True
    # Optional validators
    validate_as_sync: Union[bool, None] = None
    validate_as_matched: Union[bool, None] = None

    class Config:
        """Allow setting of partial type."""
        arbitrary_types_allowed = True

    _validate_event = validator("event", allow_reuse=True)(_ensure_list)
    _validate_duration = validator("duration", allow_reuse=True)(_ensure_list)
    _validate_n_events = validator("n_events", allow_reuse=True)(_ensure_list)
    _validate_background = validator("background", allow_reuse=True)(_ensure_list)
    _validate_gap_1 = validator("gap_1", allow_reuse=True)(_ensure_list)
    _validate_gap_2 = validator("gap_2", allow_reuse=True)(_ensure_list)
    _validate_background_weight = validator("background_weight", allow_reuse=True)(_ensure_list)
    _validate_cache = validator("cache", allow_reuse=True)(_ensure_list)
    _validate_duration_tol = validator("duration_tol", allow_reuse=True)(_ensure_list)
    _validate_normalise = validator("normalise", allow_reuse=True)(_ensure_list)

    @validator("seed", pre=True)
    def _check_if_sync_and_expected(cls, v, values, **kwargs):
        v = _ensure_list(v, values)

        v_ = []
        for s in v:
            v_.append(s if s is not None else int(datetime.now().timestamp()))

        return v_

    @validator("validate_as_matched")
    def _check_if_matched_and_expected(cls, v, values, **kwargs):
        is_matched = len(np.unique(values["n_events"])) == 1

        if v is not None:
            if not is_matched == v:
                raise IncompatibleParametersException(f"Expecting matched={is_matched}, but other params incompatible")

        return is_matched

    @validator("validate_as_sync")
    def _validate_as_sync(cls, v, values, **kwargs):
        is_matched = len(np.unique(values["n_events"])) == 1
        is_sync = is_matched & len(np.unique(values["seed"])) == 1

        if v is not None:
            if not is_sync == v:
                raise IncompatibleParametersException(f"Expecting sync={is_sync} stim, but other params incompatible")

        return is_sync

    @root_validator
    def check_same_number_of_channels(cls, values):
        lens = []
        for k in [f for f in values.keys() if f not in ['normalise_across_channels', "validate_as_sync",
                                                        "validate_as_matched", "n_channels"]]:
            lens.append(len(values[k]))

        if len(np.unique(lens)) > 1:
            raise IncompatibleParametersException(f"Number of expected channels differs between fields.")

        return values

    @root_validator
    def _check_normalisation_settings_make_sense(cls, values):
        if values['normalise_across_channels'] is True:
            if np.any(values['normalise']):
                raise IncompatibleParametersException(f"normalise_across_channels is set to True, however, at least one"
                                                      f" channel config is set to normalise during generation: "
                                                      f"{values['normalise']}")

        return values


if __name__ == "__main__":
    config = template_noisy_sine_events()

    multi_config = MultiTwoGapParams(event=config.event,
                                     n_events=12,
                                     duration=1250,
                                     background=config.background,
                                     gap_1=config.gap_1,
                                     gap_2=config.gap_2,
                                     background_weight=0.05,
                                     seed=None,
                                     cache=True,
                                     duration_tol=0.3)

    multi_config = MultiTwoGapParams(event=config.event,
                                     synchronous=True,
                                     n_events=[12, 13],
                                     duration=1250,
                                     background=config.background,
                                     gap_1=config.gap_1,
                                     gap_2=config.gap_2,
                                     background_weight=0.05,
                                     seed=123,
                                     cache=True,
                                     duration_tol=0.3)

    multi_config = MultiTwoGapParams(event=config.event,
                                     synchronous=True,
                                     n_events=12,
                                     duration=1250,
                                     background=config.background,
                                     gap_1=config.gap_1,
                                     gap_2=config.gap_2,
                                     background_weight=0.05,
                                     seed=123,
                                     cache=True,
                                     duration_tol=0.3)

    # Should raise error. TODO: Add to tests.
    multi_config = MultiTwoGapParams(event=config.event,
                                     n_events=12,
                                     duration=1250,
                                     background=config.background,
                                     gap_1=config.gap_1,
                                     gap_2=config.gap_2,
                                     background_weight=0.05,
                                     seed=None,
                                     cache=True,
                                     normalise=[False, True],
                                     duration_tol=0.3)
