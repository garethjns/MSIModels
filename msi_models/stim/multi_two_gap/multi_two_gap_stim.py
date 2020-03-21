import pickle
from typing import List, Dict, Any, Tuple

import h5py
import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from joblib import Parallel, delayed
from pydantic import PositiveInt
from tqdm import tqdm

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.two_gap.two_gap_params import TwoGapParams
from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events


class MultiTwoGapStim:
    def __init__(self, multi_params: MultiTwoGapParams):
        self.matched = multi_params.validate_as_matched
        self.synchronous = multi_params.validate_as_sync
        self.params = multi_params
        self.n_channels = self.params.n_channels

        self._individual_param_fields = ['event', 'duration', 'n_events', 'background', 'gap_1', 'gap_2',
                                          'background_weight', 'seed', 'cache', 'duration_tol']
        self._y = None
        self._y_mask = None

    @classmethod
    def template(cls) -> "MultiTwoGapStim":
        config = template_noisy_sine_events()

        multi_params = MultiTwoGapParams(event=config.event,
                                         n_events=[12, 12],
                                         duration=1250,
                                         background=config.background,
                                         gap_1=config.gap_1,
                                         gap_2=config.gap_2,
                                         background_weight=0.05,
                                         seed=123,
                                         cache=True,
                                         duration_tol=0.5)

        return MultiTwoGapStim(multi_params)

    def _generate_channels(self):
        # Create configs
        self.individual_chan_params = []
        for c in range(self.n_channels):
            self.individual_chan_params.append(
                TwoGapParams(**{k: self.params.dict()[k][c] for k in self._individual_param_fields}))

        self.channels = [TwoGapStim(c) for c in self.individual_chan_params]

    def _generate(self):
        self._generate_channels()
        return [c.y for c in self.channels], [c.y_mask for c in self.channels]

    def _get_or_generate(self) -> Tuple[TwoGapStim, TwoGapStim]:
        if (self._y is None) or (self._y_mask is None):
            y, y_mask = self._generate()
        else:
            y = self._y
            y_mask = self._y_mask

        if self.params.cache:
            self._y = y
            self._y_true = y_mask

        return y, y_mask

    @property
    def y(self):
        return self._get_or_generate()[0]

    @property
    def y_true(self):
        return self._get_or_generate()[1]

    @classmethod
    def generate(cls, config_left: TwoGapParams,
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
                      n_jobs=n_jobs)(delayed(MultiTwoGapStim._batch)(n=batch_size,
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

    @classmethod
    def _batch(cls, config_left: TwoGapParams,
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
        agg_y_dec = np.zeros(shape=(n, 2))
        agg_y_rate = np.zeros(shape=(n,))

        for n_i in range(n):
            n_events = np.random.choice(events)
            config_left.n_events = n_events
            config_right.n_events = n_events

            # Match for now
            agg_y_rate[n_i] = n_events
            agg_y_dec[n_i, int(n_events >= events_mean)] = 1

            left_stim = TwoGapStim(config_left)
            right_stim = TwoGapStim(config_right)

            left_configs.append(pickle.dumps(config_left))
            left_x[n_i, :] = left_stim.y.y
            left_x_indicators[n_i, :] = left_stim.y_mask.y
            right_configs.append(pickle.dumps(config_right))
            right_x[n_i, :] = right_stim.y.y
            right_x_indicators[n_i, :] = right_stim.y_mask.y

        return {'left': {'x': np.expand_dims(left_x, axis=2),
                         'x_mask': np.expand_dims(left_x_indicators, axis=2),
                         'y_rate': agg_y_rate,
                         'y_dec': agg_y_dec,
                         'configs': left_configs},
                'right': {'x': np.expand_dims(right_x, axis=2),
                          'x_mask': np.expand_dims(right_x_indicators, axis=2),
                          'y_rate': agg_y_rate,
                          'y_dec': agg_y_dec,
                          'configs': right_configs},
                'agg': {'y_rate': agg_y_rate,
                        'y_dec': agg_y_dec}}


if __name__ == "__main__":
    multi_stim = MultiTwoGapStim.template()
    multi_stim.y
