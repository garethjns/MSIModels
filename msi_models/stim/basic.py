import numpy as np
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.envelopes.envelope import Envelope


def template_noisy_sine(duration: int = 1000,
                        fs: int = 5000,
                        freq: int = 12,
                        noise_mag: float = -80) -> CompoundComponent:
    """

    :param: fs: Sampling rate, Hz.
    :param duration: Duration in ms.
    :param freq: Frequency of sine wave, Hz.
    :param noise_mag: Noise mag in db (relative to sine mag of 1v)
    :return:
    """
    sin = SineComponent(freq=freq,
                        mag=1,
                        fs=fs,
                        duration=duration)
    noise = NoiseComponent(fs=fs,
                           duration=duration,
                           mag=db_to_lin(ref=1,
                                         db_change=noise_mag))

    return CompoundComponent([sin, noise])


def template_complex():
    """
    Sine waves of increasing complexity in 3 steps:
    4 Hz -> 4 + (6 and 2) ->  4 + (6 and 2) + (8 and 10)

    Overlayed with increasing noise.
    :return:
    """
    duration = 1000

    start = 0
    sine_4 = SineComponent(start=start,
                           duration=duration - start,
                           freq=4)

    start = 200
    sine_2_6 = CompoundComponent([SineComponent(start=start,
                                                duration=duration - start,
                                                freq=2),
                                  SineComponent(start=start,
                                                duration=duration - start,
                                                freq=6)])

    start = 600
    sine_8_10 = CompoundComponent([SineComponent(start=start,
                                                 duration=duration - start,
                                                 freq=8),
                                   SineComponent(start=start,
                                                 duration=duration - start,
                                                 freq=10)])

    class IncreasingEnvelope(Envelope):
        def f(self, y):
            return y * np.linspace(0, 1, len(y))

    noise = NoiseComponent(start=0,
                           duration=1000,
                           envelope=IncreasingEnvelope,
                           mag=db_to_lin(ref=1,
                                         db_change=-120))

    return CompoundComponent([sine_4, sine_2_6, sine_8_10, noise])


if __name__ == "__main__":
    ev = template_noisy_sine()
    ev.plot(show=True,
            channels=True)

    ev = template_complex()
    ev.plot(show=True)
    ev.plot_subplots(show=True)

    ev.to_list()
