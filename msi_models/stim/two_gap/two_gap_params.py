from functools import partial
from typing import Union, Callable

import numpy as np
from audiodag.signal.components.component import Component
from pydantic import BaseModel, PositiveInt, validator, root_validator

from msi_models.exceptions.params import InvalidParameterException, IncompatibleParametersException


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
    :param normalise: Whether or not to normalise the magnitude of the stim to between 0 -> 1
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
    normalise: bool = False

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
