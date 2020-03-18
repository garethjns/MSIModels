from typing import List, Tuple

import h5py
import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from joblib import Parallel, delayed
from pydantic import PositiveInt
from tqdm import tqdm

from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_params import TwoGapParams


def batch(config: TwoGapParams,
          n: PositiveInt = 400,
          fs: PositiveInt = 500,
          events: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if events is None:
        events = [11, 12, 13, 14, 15, 16]
    events_mean = np.mean(events)

    x_indicators = np.zeros(shape=(n, ms_to_pts(config.duration, fs)))
    x = np.zeros(shape=(n, ms_to_pts(config.duration, fs)))
    y_dec = np.zeros(shape=(n, 2))
    y_rate = np.zeros(shape=(n,))

    for n_i in range(n):
        n_events = np.random.choice(events)
        config.n_events = n_events
        y_rate[n_i] = n_events

        y_dec[n_i, int(n_events >= events_mean)] = 1

        stim = TwoGapStim(config)
        x[n_i, :] = stim.y.y
        x_indicators[n_i, :] = stim.y_true.y

    return np.expand_dims(x, axis=2), np.expand_dims(x_indicators, axis=2), y_rate, y_dec


def generate_unisensory_binary(config: TwoGapParams,
                               n: PositiveInt = 400,
                               fs: PositiveInt = 500,
                               batch_size: PositiveInt = 20,
                               events: List[int] = None,
                               fn: str = 'unisensory_data.hdf5',
                               n_jobs: int = -2):
    n_batches = int(n / batch_size)
    xy = Parallel(backend='loky',
                  verbose=0,
                  n_jobs=n_jobs)(delayed(batch)(n=batch_size,
                                                config=config,
                                                fs=fs,
                                                events=events) for _ in tqdm(range(n_batches)))

    x = np.concatenate([b[0] for b in xy], axis=0)
    x_indicators = np.concatenate([b[1] for b in xy], axis=0)
    y_rate = np.concatenate([b[2] for b in xy], axis=0)
    y_dec = np.concatenate([b[3] for b in xy], axis=0)

    with h5py.File(fn, 'w') as f:
        f.create_dataset("x", data=x, compression='gzip',
                         chunks=(100, config.duration / 1000 * fs, 1))
        f.create_dataset("x_mask", data=x_indicators, compression='gzip',
                         chunks=(100, config.duration / 1000 * fs, 1))
        f.create_dataset("y_rate", data=y_rate, compression='gzip',
                         chunks=(100,))
        f.create_dataset("y_dec", data=y_dec, compression='gzip',
                         chunks=(100, 2))


if __name__ == "__main__":
    from msi_models.stim.two_gap.two_gap_templates import template_sine_events, template_noisy_sine_events

    generate_unisensory_binary(config=template_noisy_sine_events(duration=1300,
                                                                 fs=500,
                                                                 background_mag=0.09,
                                                                 duration_tol=0.5),
                               n=3000,
                               batch_size=5,
                               fn='unisensory_data_hard.hdf5',
                               n_jobs=-2)
