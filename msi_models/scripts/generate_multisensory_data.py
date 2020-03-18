import gc
from typing import List, Dict, Any

import h5py
import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from joblib import Parallel, delayed
from pydantic import PositiveInt
from tqdm import tqdm

from msi_models.stim.two_gap.two_gap_params import TwoGapParams
from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
import pickle


def multisensory_batch(config_left: TwoGapParams,
                       config_right: TwoGapParams,
                       n: PositiveInt = 400,
                       fs: PositiveInt = 500,
                       events: List[int] = None) -> Dict[str, Dict[str, Any]]:
    if events is None:
        events = [11, 12, 13, 14, 15, 16]
    events_mean = np.mean(events)

    left_x_indicators = np.zeros(shape=(n, ms_to_pts(config_left.duration, fs)))
    left_x = np.zeros(shape=(n, ms_to_pts(config_left.duration, fs)))
    left_configs = []
    right_x_indicators = np.zeros(shape=(n, ms_to_pts(config_right.duration, fs)))
    right_x = np.zeros(shape=(n, ms_to_pts(config_right.duration, fs)))
    right_configs = []
    y_dec = np.zeros(shape=(n, 2))
    y_rate = np.zeros(shape=(n,))

    for n_i in range(n):
        n_events = np.random.choice(events)
        config_left.n_events = n_events
        config_right.n_events = n_events

        y_rate[n_i] = n_events

        y_dec[n_i, int(n_events >= events_mean)] = 1

        left_stim = TwoGapStim(config_left)
        right_stim = TwoGapStim(config_right)

        left_configs.append(pickle.dumps(config_left))
        left_x[n_i, :] = left_stim.y.y
        left_x_indicators[n_i, :] = left_stim.y_true.y
        right_configs.append(pickle.dumps(config_right))
        right_x[n_i, :] = right_stim.y.y
        right_x_indicators[n_i, :] = right_stim.y_true.y

    return {'left': {'x': np.expand_dims(left_x, axis=2),
                     'x_mask': np.expand_dims(left_x_indicators, axis=2),
                     'y_rate': y_rate,
                     'y_dec': y_dec,
                     'configs': left_configs},
            'right': {'x': np.expand_dims(right_x, axis=2),
                      'x_mask': np.expand_dims(right_x_indicators, axis=2),
                      'y_rate': y_rate,
                      'y_dec': y_dec,
                      'configs': right_configs}}


def generate_multisensory(config_left: TwoGapParams,
                          config_right: TwoGapParams,
                          n: PositiveInt = 400,
                          fs: PositiveInt = 500,
                          batch_size: PositiveInt = 20,
                          events: List[int] = None,
                          fn: str = 'multisensory_data.hdf5',
                          n_jobs: int = -2):
    n_batches = int(n / batch_size)
    xy = Parallel(backend='loky',
                  verbose=0,
                  n_jobs=n_jobs)(delayed(multisensory_batch)(n=batch_size,
                                                             config_left=config_left,
                                                             config_right=config_right,
                                                             fs=fs,
                                                             events=events) for _ in tqdm(range(n_batches)))

    with h5py.File(fn, 'w') as f:
        for channel_key in xy[0].keys():
            for content_key in xy[0][channel_key].keys():
                concat_output = np.concatenate([b[channel_key][content_key] for b in xy],
                                               axis=0)

                f.create_dataset("/".join([channel_key, content_key]),
                                 data=concat_output,
                                 compression='gzip')


if __name__ == "__main__":
    from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events, template_sine_events

    common_config_kwargs = {"duration": 1300,
                            "fs": 500,
                            "background_mag": 0.09,
                            "duration_tol": 0.5}

    generate_multisensory(config_left=template_noisy_sine_events(**common_config_kwargs),
                          config_right=template_noisy_sine_events(**common_config_kwargs),
                          n=2000,
                          batch_size=10,
                          fn='multisensory_data.hdf5',
                          n_jobs=-2)
