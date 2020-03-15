from typing import List, Tuple

import h5py
import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from joblib import Parallel, delayed
from pydantic import PositiveInt
from tqdm import tqdm

from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events


def batch(n: PositiveInt = 400,
          fs: PositiveInt = 512,
          duration: PositiveInt = 1000,
          events: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if events is None:
        events = [11, 12, 13, 14, 15, 16]
    events_mean = np.mean(events)

    x_indicators = np.zeros(shape=(n, ms_to_pts(duration, fs)))
    x = np.zeros(shape=(n, ms_to_pts(duration, fs)))
    y_dec = np.zeros(shape=(n, 2))
    y_rate = np.zeros(shape=(n,))

    for n_i in range(n):
        n_events = np.random.choice(events)
        y_rate[n_i] = n_events

        y_dec[n_i, int(n_events >= events_mean)] = 1

        params = template_sine_events(n_events=n_events,
                                      duration_tol=0.8,
                                      duration=duration,
                                      fs=fs)
        stim = TwoGapStim(params)
        x[n_i, :] = stim.y.y
        x_indicators[n_i, :] = stim.y_true.y

    return np.expand_dims(x, axis=2), np.expand_dims(x_indicators, axis=2), y_rate, y_dec


def generate_unisensory_binary(n: PositiveInt = 400,
                               fs: PositiveInt = 500,
                               duration: PositiveInt = 1300,
                               batch_size: PositiveInt = 20,
                               events: List[int] = None,
                               fn: str = 'unisensory_data.hdf5',
                               n_jobs: int = -2):
    n_batches = int(n / batch_size)
    xy = Parallel(backend='loky',
                  verbose=0,
                  n_jobs=n_jobs)(delayed(batch)(n=batch_size,
                                                duration=duration,
                                                fs=fs,
                                                events=events) for _ in tqdm(range(n_batches)))

    x = np.concatenate([b[0] for b in xy], axis=0)
    x_indicators = np.concatenate([b[1] for b in xy], axis=0)
    y_rate = np.concatenate([b[2] for b in xy], axis=0)
    y_dec = np.concatenate([b[3] for b in xy], axis=0)

    with h5py.File(fn, 'w') as f:
        f.create_dataset("x_1", data=x, compression='gzip', chunks=(100, duration / 1000 * fs, 1))
        f.create_dataset("x_indicators", data=x_indicators, compression='gzip', chunks=(100, duration / 1000 * fs, 1))
        f.create_dataset("rate_output", data=y_rate, compression='gzip', chunks=(100,))
        f.create_dataset("dec_output", data=y_dec, compression='gzip', chunks=(100, 2))


if __name__ == "__main__":
    generate_unisensory_binary(n=3000,
                               batch_size=20)
