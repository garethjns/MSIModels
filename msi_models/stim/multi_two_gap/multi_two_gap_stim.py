import pickle
from typing import List, Dict, Any, Tuple, Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from audiodag.signal.digital.conversion import ms_to_pts
from audiodag.signal.components.component import CompoundComponent
from joblib import Parallel, delayed
from pydantic import PositiveInt
from tqdm import tqdm

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.two_gap.two_gap_params import TwoGapParams
from msi_models.stim.two_gap.two_gap_stim import TwoGapStim


class MultiTwoGapStim:
    def __init__(self, multi_params: MultiTwoGapParams):
        self.matched = multi_params.validate_as_matched
        self.synchronous = multi_params.validate_as_sync
        self.params = multi_params
        self.n_channels = self.params.n_channels

        self.channel_params: List[TwoGapParams]
        self._individual_param_fields = ['event', 'duration', 'n_events', 'background', 'gap_1', 'gap_2',
                                         'background_weight', 'seed', 'cache', 'duration_tol']
        self.channels: List[TwoGapStim]
        self._generate_channels()

        self._y = None
        self._y_mask = None

    def _generate_channels(self):
        # Create configs
        self.channel_params = []
        for c in range(self.n_channels):
            self.channel_params.append(
                TwoGapParams(**{k: self.params.dict()[k][c] for k in self._individual_param_fields}))

        self.channels = [TwoGapStim(c) for c in self.channel_params]

    def _generate(self):
        self._generate_channels()
        return [c.y for c in self.channels], [c.y_mask for c in self.channels]

    def _get_or_generate(self) -> Tuple[List[TwoGapStim], List[TwoGapStim]]:
        if (self._y is None) or (self._y_mask is None):
            y, y_mask = self._generate()
        else:
            y = self._y
            y_mask = self._y_mask

        if self.params.cache:
            self._y = y
            self._y_mask = y_mask

        return y, y_mask

    @property
    def y(self) -> List[TwoGapStim]:
        return self._get_or_generate()[0]

    @property
    def y_mask(self) -> List[TwoGapStim]:
        return self._get_or_generate()[1]

    def plot(self,
             show: bool = False):
        fig, ax = plt.subplots(nrows=self.n_channels,
                               ncols=1)

        for ax_, c in zip(ax, self.channels):
            plt.sca(ax_)
            c.y.plot(show=False)
            c.y_mask.plot(show=False)

        if show:
            plt.show()

    @classmethod
    def generate(cls, template: Callable,
                 n: int = 400,
                 batch_size: int = 20,
                 events: List[int] = None,
                 fs: int = 500,
                 fn: str = 'multisensory_data.hdf5',
                 n_jobs: int = -2,
                 template_kwargs: Dict[str, Any] = None):
        """Supports 2 channels for now."""

        n_batches = int(n / batch_size)
        xy = Parallel(backend='loky',
                      verbose=0,
                      n_jobs=n_jobs)(delayed(MultiTwoGapStim._batch)(n=batch_size,
                                                                     template=template,
                                                                     fs=fs,
                                                                     events=events,
                                                                     template_kwargs=template_kwargs) for _ in
                                     tqdm(range(n_batches)))

        with h5py.File(fn, 'w') as f:
            for channel_key in xy[0].keys():
                for content_key in xy[0][channel_key].keys():
                    concat_output = np.concatenate([b[channel_key][content_key] for b in xy],
                                                   axis=0)

                    f.create_dataset("/".join([channel_key, content_key]),
                                     data=concat_output,
                                     compression='gzip')

    @classmethod
    def _batch(cls, template: Callable,
               n: PositiveInt = 20,
               fs: int = 500,
               events: List[int] = None,
               template_kwargs: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:

        if template_kwargs is None:
            template_kwargs = {}

        if events is None:
            events = [11, 12, 13, 14, 15, 16]

        events_mean = np.mean(events)
        example_params = template(fs=fs, **template_kwargs)

        left_x_indicators = np.zeros(shape=(n, ms_to_pts(example_params.duration[0], fs)))
        left_x = np.zeros(shape=(n, ms_to_pts(example_params.duration[0], fs)))
        left_configs = []
        left_y_rate = np.zeros(shape=(n,))
        left_y_dec = np.zeros(shape=(n, 2))

        right_x_indicators = np.zeros(shape=(n, ms_to_pts(example_params.duration[1], fs)))
        right_x = np.zeros(shape=(n, ms_to_pts(example_params.duration[1], fs)))
        right_configs = []
        right_y_dec = np.zeros(shape=(n, 2))
        right_y_rate = np.zeros(shape=(n,))

        agg_y_dec = np.zeros(shape=(n, 2))
        agg_y_rate = np.zeros(shape=(n,))

        for n_i in range(n):
            multi_stim = MultiTwoGapStim(template(fs=fs, **template_kwargs))

            left_y_rate[n_i] = multi_stim.channel_params[0].n_events
            right_y_rate[n_i] = multi_stim.channel_params[1].n_events
            agg_y_rate[n_i] = np.mean([left_y_rate[n_i], right_y_rate[n_i]])

            left_y_dec[n_i, int(left_y_rate[n_i] >= events_mean)] = 1
            right_y_dec[n_i, int(right_y_rate[n_i] >= events_mean)] = 1
            agg_y_dec[n_i, int(agg_y_rate[n_i] >= events_mean)] = 1

            left_configs.append(pickle.dumps(multi_stim.channel_params[0]))
            left_x[n_i, :] = multi_stim.y[0].y
            left_x_indicators[n_i, :] = multi_stim.y_mask[0].y
            right_configs.append(pickle.dumps(multi_stim.channel_params[1]))
            right_x[n_i, :] = multi_stim.y[1].y
            right_x_indicators[n_i, :] = multi_stim.y_mask[1].y

        return {'left': {'x': np.expand_dims(left_x, axis=2),
                         'x_mask': np.expand_dims(left_x_indicators, axis=2),
                         'y_rate': left_y_rate,
                         'y_dec': left_y_dec,
                         'configs': left_configs},
                'right': {'x': np.expand_dims(right_x, axis=2),
                          'x_mask': np.expand_dims(right_x_indicators, axis=2),
                          'y_rate': right_y_rate,
                          'y_dec': right_y_dec,
                          'configs': right_configs},
                'agg': {'y_rate': agg_y_rate,
                        'y_dec': agg_y_dec}}


if __name__ == "__main__":
    from msi_models.stim.multi_two_gap.multi_two_gap_templates import (template_matched, template_sync,
                                                                       template_unmatched)

    multi_stim_unmatched = MultiTwoGapStim(template_unmatched(cache=False))
    multi_stim_unmatched.y
    multi_stim_unmatched.plot(show=True)

    multi_stim_matched = MultiTwoGapStim(template_matched(cache=True))
    multi_stim_matched.y
    multi_stim_matched.plot(show=True)

    multi_stim_sync = MultiTwoGapStim(template_sync(cache=True))
    multi_stim_sync.y
    multi_stim_sync.plot(show=True)

    MultiTwoGapStim.generate(template_sync,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})
