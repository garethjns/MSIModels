import unittest
from functools import partial

from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent
from audiodag.signal.envelopes.templates import CosRiseEnvelope

from msi_models.exceptions.params import InvalidParameterException, IncompatibleParametersException
from msi_models.stim.two_gap.two_gap_params import TwoGapParams


class TestTwoGapParams(unittest.TestCase):
    _sut = TwoGapParams

    def test_invalid_background_param_raises_validation_error(self):
        event = partial(SineComponent, duration=20, freq=8, fs=1000, mag=2)
        gap_1 = partial(NoiseComponent, duration=20, mag=0.02, fs=1200)
        background = partial(NoiseComponent, duration=1000, fs=1000, envelope=CosRiseEnvelope(fs=1200, rise=20))

        self.assertRaises(InvalidParameterException,
                          lambda: TwoGapParams(duration=1000, n_events=10, event=event, gap_1=gap_1, gap_2=gap_1,
                                               background=background))

    def test_incompatible_fs_between_partials_raise_validation_error(self):
        event = partial(SineComponent, duration=20, freq=8, fs=1000, mag=2)
        gap_1 = partial(NoiseComponent, duration=20, mag=0.02, fs=1200)
        background = partial(NoiseComponent, duration=1000, fs=20, mag=0.02,
                             envelope=partial(CosRiseEnvelope, rise=20))

        self.assertRaises(IncompatibleParametersException,
                          lambda: TwoGapParams(duration=1000, n_events=10, event=event, gap_1=gap_1, gap_2=gap_1,
                                               background=background))
