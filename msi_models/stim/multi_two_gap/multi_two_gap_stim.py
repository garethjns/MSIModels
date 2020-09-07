import pickle
from typing import List, Dict, Any, Tuple, Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audiodag.signal.digital.conversion import ms_to_pts
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
                                         'background_weight', 'seed', 'cache', 'duration_tol', 'normalise']
        self.channels: List[TwoGapStim]
        self._generate_channels()

        self._y = None
        self._y_mask = None

    def _generate_channels(self) -> None:
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
    def y_objs(self) -> List[TwoGapStim]:
        return self._get_or_generate()[0]

    @staticmethod
    def _normalise(y: np.ndarray) -> np.ndarray:
        y_min = y.min()
        return (y - y_min) / (y.max() - y_min)

    @property
    def y(self) -> np.ndarray:
        if self.params.normalise_across_channels:
            return self._normalise(np.array([c.y for c in self._get_or_generate()[0]]))
        else:
            return np.array([c.y for c in self._get_or_generate()[0]])

    @property
    def y_mask(self) -> np.ndarray:
        if self.params.normalise_across_channels:
            return self._normalise(np.array([c.y for c in self._get_or_generate()[1]]))
        else:
            return np.array([c.y for c in self._get_or_generate()[1]])

    @property
    def y_mask_objs(self) -> List[TwoGapStim]:
        return self._get_or_generate()[1]

    def plot(self,
             show: bool = False):
        fig, ax = plt.subplots(nrows=self.n_channels,
                               ncols=1)

        for c, ax_ in enumerate(ax):
            ax_.plot(self.y[c, :])
            ax_.plot(self.y_mask[c, :])

        if show:
            plt.show()

    @classmethod
    def generate(cls, templates: List[Callable], n: int = 400, batch_size: int = 20, events: List[int] = None,
                 fs: int = 500, fn: str = 'multisensory_data.hdf5', n_jobs: int = -2,
                 template_kwargs: Dict[str, Any] = None):
        """Supports 2 channels for now."""

        n_batches = int(n / batch_size)
        batch_templates = np.random.choice(templates, replace=True, size=n_batches)

        xy = Parallel(verbose=1,
                      n_jobs=n_jobs)(delayed(MultiTwoGapStim._batch)(n=batch_size, template=t, fs=fs,
                                                                     events=events, template_kwargs=template_kwargs)
                                     for t in tqdm(batch_templates))

        with h5py.File(fn, 'w') as f:
            for channel_key in xy[0].keys():
                if channel_key != 'summary':
                    for content_key in xy[0][channel_key].keys():
                        f.create_dataset("/".join([channel_key, content_key]),
                                         data=np.concatenate([b[channel_key][content_key] for b in xy], axis=0),
                                         compression='gzip')

        summary = pd.concat([b["summary"] for b in xy], axis=0).reset_index(drop=True)
        summary.to_hdf(fn, key='summary', mode='a')

    @classmethod
    def _batch(cls, template: "MultiTwoGapTemplate",
               n: PositiveInt = 20,
               fs: int = 500,
               events: List[int] = None,
               template_kwargs: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:

        if template_kwargs is None:
            template_kwargs = {}

        if events is None:
            events = [11, 12, 13, 14, 15, 16]

        events_mean = np.mean(events)
        mtg_template = template.set_options(n_events=events, fs=fs, **template_kwargs)
        example_params = mtg_template.build()

        left_x_indicators = np.zeros(shape=(n, ms_to_pts(example_params.duration[0], fs)),
                                     dtype=np.float32)
        left_x = np.zeros(shape=(n, ms_to_pts(example_params.duration[0], fs)),
                          dtype=np.float32)
        left_configs = []
        left_y_rate = np.zeros(shape=(n,), dtype=np.uint16)
        left_y_dec = np.zeros(shape=(n, 2), dtype=np.float32)

        right_x_indicators = np.zeros(shape=(n, ms_to_pts(example_params.duration[1], fs)), dtype=np.float32)
        right_x = np.zeros(shape=(n, ms_to_pts(example_params.duration[1], fs)), dtype=np.float32)
        right_configs = []
        right_y_dec = np.zeros(shape=(n, 2), dtype=np.float32)
        right_y_rate = np.zeros(shape=(n,), dtype=np.uint16)

        agg_y_dec = np.zeros(shape=(n, 2), dtype=np.float32)
        agg_y_rate = np.zeros(shape=(n,), dtype=np.uint16)

        multi_stim_configs = []
        for n_i in range(n):
            multi_stim_config = mtg_template.build()
            multi_stim_configs.append(multi_stim_config)
            multi_stim = MultiTwoGapStim(multi_stim_config)

            if template.name == "left_only":
                # Unisensory left
                left_y_rate[n_i] = multi_stim.channel_params[0].n_events
                right_y_rate[n_i] = multi_stim.channel_params[0].n_events
            elif template.name == "right_only":
                # Unisensory right
                left_y_rate[n_i] = multi_stim.channel_params[1].n_events
                right_y_rate[n_i] = multi_stim.channel_params[1].n_events
            else:
                # Multisensory
                left_y_rate[n_i] = multi_stim.channel_params[0].n_events
                right_y_rate[n_i] = multi_stim.channel_params[1].n_events

            agg_y_rate[n_i] = np.mean([left_y_rate[n_i], right_y_rate[n_i]])

            left_y_dec[n_i, int(left_y_rate[n_i] >= events_mean)] = 1
            right_y_dec[n_i, int(right_y_rate[n_i] >= events_mean)] = 1
            agg_y_dec[n_i, int(agg_y_rate[n_i] >= events_mean)] = 1

            left_configs.append(pickle.dumps(multi_stim.channel_params[0]))
            left_x[n_i, :] = multi_stim.y_objs[0].y
            left_x_indicators[n_i, :] = multi_stim.y_mask_objs[0].y
            right_configs.append(pickle.dumps(multi_stim.channel_params[1]))
            right_x[n_i, :] = multi_stim.y_objs[1].y
            right_x_indicators[n_i, :] = multi_stim.y_mask_objs[1].y

        summary = pd.DataFrame({
            'type': [template.value] * len(multi_stim_configs),
            'n_channels': [cfg.dict()['n_channels'] for cfg in multi_stim_configs],
            'left_duration': [cfg.dict()['duration'][0] for cfg in multi_stim_configs],
            'right_duration': [cfg.dict()['duration'][1] for cfg in multi_stim_configs],
            'left_n_events': [cfg.dict()['n_events'][0] for cfg in multi_stim_configs],
            'right_n_events': [cfg.dict()['n_events'][1] for cfg in multi_stim_configs],
            'left_background_weight': [cfg.dict()['background_weight'][0] for cfg in multi_stim_configs],
            'right_background_weight': [cfg.dict()['background_weight'][1] for cfg in multi_stim_configs],
            'left_seed': [cfg.dict()['seed'][0] for cfg in multi_stim_configs],
            'right_seed': [cfg.dict()['seed'][1] for cfg in multi_stim_configs],
            'left_duration_tol': [cfg.dict()['duration_tol'][0] for cfg in multi_stim_configs],
            'right_duration_tol': [cfg.dict()['duration_tol'][1] for cfg in multi_stim_configs],
            'left_normalise': [cfg.dict()['normalise'][0] for cfg in multi_stim_configs],
            'right_normalise': [cfg.dict()['normalise'][1] for cfg in multi_stim_configs],
            'sync': [cfg.dict()['validate_as_sync'] for cfg in multi_stim_configs],
            'matched': [cfg.dict()['validate_as_matched'] for cfg in multi_stim_configs]})

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
                        'y_dec': agg_y_dec},
                'summary': summary}


if __name__ == "__main__":
    from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate

    multi_stim_unmatched = MultiTwoGapStim(MultiTwoGapTemplate(1).build())

    multi_stim_unmatched.y
    multi_stim_unmatched.plot(show=True)

    multi_stim_matched = MultiTwoGapStim(template_matched(cache=True))
    multi_stim_matched.y_objs
    multi_stim_matched.plot(show=True)

    multi_stim_sync = MultiTwoGapStim(template_sync(cache=True))
    multi_stim_sync.y_objs
    multi_stim_sync.plot(show=True)

    MultiTwoGapStim.generate(template_sync,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})
