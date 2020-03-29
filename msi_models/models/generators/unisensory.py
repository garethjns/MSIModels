from typing import List

import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from pydantic import PositiveInt

from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events


def unisensory_binary(n: PositiveInt = 2,
                      fs: PositiveInt = 512,
                      duration: PositiveInt = 1300,
                      duration_tol: float = 0.5,
                      events: List[int] = None):
    while True:

        if events is None:
            events = [8, 9, 10, 11, 12, 13]
        events_mean = np.mean(events)

        x = np.zeros(shape=(n, ms_to_pts(duration, fs)))
        y_dec = np.zeros(shape=(n, 2))
        y_rate = np.zeros(shape=(n,))

        for n_i in range(n):
            n_events = np.random.choice(events)
            y_rate[n_i] = n_events

            y_dec[n_i, int(n_events >= events_mean)] = 1

            params = template_sine_events(duration=duration,
                                          duration_tol=duration_tol,
                                          fs=fs)
            x[n_i, :] = TwoGapStim(params).y.y

        yield {"input_1": np.expand_dims(x, axis=2)}, {"rate_output": y_rate,
                                                       "dec_output": y_dec}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y = next(unisensory_binary(n=5))
    plt.plot(x["input_1"][0, :])
    plt.show()
