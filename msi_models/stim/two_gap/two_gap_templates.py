from functools import partial

from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.tonal_component import SineComponent
from audiodag.signal.envelopes.templates import CosRiseEnvelope
from pydantic import PositiveInt

from msi_models.stim.two_gap.two_gap_params import TwoGapParams


def _noise_gaps(gap_1_duration: PositiveInt = 25,
                gap_2_duration: PositiveInt = 50,
                fs: PositiveInt = 1000):
    gap_1 = partial(NoiseComponent, duration=gap_1_duration, mag=0.02, fs=fs)
    gap_2 = partial(NoiseComponent, duration=gap_2_duration, mag=0.02, fs=fs)

    return gap_1, gap_2


def template_sine_events(duration: PositiveInt = 1300,
                         n_events: PositiveInt = 13,
                         fs: PositiveInt = 1000,
                         gap_1_duration: PositiveInt = 25,
                         gap_2_duration: PositiveInt = 50,
                         background_mag: float = 0.02,
                         duration_tol: float = 0.5,
                         normalise: bool = True,
                         **kwargs) -> "TwoGapParams":
    """This example has an SineComponent event."""
    event = partial(SineComponent, duration=gap_1_duration, freq=2, fs=fs, mag=2, cache=True)

    gap_1, gap_2 = _noise_gaps(gap_1_duration, gap_2_duration, fs=fs)
    background = partial(NoiseComponent, duration=duration, fs=fs, mag=background_mag,
                         envelope=partial(CosRiseEnvelope, rise=20), cache=True)

    return TwoGapParams(duration=duration, n_events=n_events, event=event, gap_1=gap_1, gap_2=gap_2,
                        background=background, background_weight=2, duration_tol=duration_tol,
                        normalise=normalise, **kwargs)


def template_noisy_sine_events(duration: PositiveInt = 1300,
                               n_events: PositiveInt = 13,
                               fs: PositiveInt = 1000,
                               gap_1_duration: PositiveInt = 25,
                               gap_2_duration: PositiveInt = 50,
                               background_mag: float = 0.02,
                               duration_tol: float = 0.5,
                               normalise: bool = True,
                               **kwargs) -> "TwoGapParams":
    """This example has an CompoundComponent event that comprises a sine component and a noise component."""

    event_sine_component = SineComponent(duration=gap_1_duration, freq=8, fs=fs, mag=2, cache=True)
    event_noise_component = NoiseComponent(duration=gap_1_duration, fs=fs, mag=0.02, cache=True)
    event = partial(CompoundComponent,
                    events=[event_sine_component, event_noise_component], cache=True)

    gap_1, gap_2 = _noise_gaps(gap_1_duration, gap_2_duration, fs=fs)
    background = partial(NoiseComponent, duration=duration, fs=fs, mag=background_mag,
                         envelope=partial(CosRiseEnvelope, rise=20), cache=True)

    return TwoGapParams(duration=duration, n_events=n_events, event=event, gap_1=gap_1, gap_2=gap_2,
                        background=background, background_weight=2, duration_tol=duration_tol,
                        normalise=normalise, **kwargs)
