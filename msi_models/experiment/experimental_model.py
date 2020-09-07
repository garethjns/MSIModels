import gc
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.models.conv.unisensory_templates import UnisensoryClassifier
from msi_models.models.keras_sk_base import KerasSKBase
from msi_models.stimset.multi_channel import MultiChannel


@dataclass
class ExperimentalModel:
    model: Union[UnisensoryClassifier, MultisensoryClassifier]
    name: str = "unnamed_model"

    def __init__(self, model: KerasSKBase,
                 name: str = "unnamed_model"):
        self.name = name
        self.model = model
        self.preds_train: Union[None, Dict[str, np.ndarray]] = None
        self.preds_test: Union[None, Dict[str, np.ndarray]] = None

        self.run_id: int
        self.results: pd.DataFrame = pd.DataFrame()

    def fit(self, data: MultiChannel,
            validation_split: float = 0.4, **kwargs):
        model_outputs = self.model._loss_weights

        self.model.fit(data.x_train, {k: data.y_train[k] for k in model_outputs},
                       # Only input ys used in losses
                       validation_split=validation_split, shuffle=True,
                       **kwargs)

    def predict(self, data: MultiChannel) -> Tuple[Dict[str, np.ndarray],
                                                   Dict[str, np.ndarray]]:

        if self.preds_train is None:
            self.preds_train = self._predict_batches(data.x_train)

        if self.preds_test is None:
            self.preds_test = self._predict_batches(data.x_test)

        return self.preds_train, self.preds_test

    def _predict_batches(self, data: Dict[str, np.ndarray],
                         chunk_size: int = 1000) -> Dict[str, np.ndarray]:

        n = len(list(data.values())[0])
        n_chunks = int(np.ceil(n / chunk_size))
        split_x = {k: np.array_split(v, n_chunks) for k, v in data.items()}
        preds = []
        for chunk_i in range(n_chunks):
            x = {k: v[chunk_i] for k, v in split_x.items()}
            preds.append(self.model.predict_dict(x))

        del split_x, x
        gc.collect()

        concat_preds = {}
        # NB: List here to force copy of dict keys so it can be popped from in loop
        for k in list(preds[0].keys()):
            concat_preds[k] = np.concatenate([p.pop(k) for p in preds], axis=0)
            gc.collect()

        return concat_preds

    def calc_prop_fast(self, data: MultiChannel,
                       type_key: str = 'type', rate_key: str = 'agg_y_rate') -> List[pd.DataFrame]:
        train_df, test_df = self.report(data)

        return [df[[type_key, rate_key, 'preds_dec']].groupby([rate_key, type_key]).mean().reset_index(drop=False)
                for df in [train_df, test_df]]

    def plot_prop_fast(self, data: MultiChannel, type_key='type', rate_key: str = 'agg_y_rate'):
        train_pf, test_pf = self.calc_prop_fast(data, type_key=type_key, rate_key=rate_key)

        rates = train_pf[rate_key].unique()
        typs = np.sort(data.summary.type.unique())
        n_subplots = len(typs)
        fig, axs = plt.subplots(ncols=n_subplots, figsize=(2.5 * n_subplots, 8))
        for ai, (ax, ty) in enumerate(zip(axs, typs)):
            for name, df in zip(['train', 'test'], [train_pf, test_pf]):
                df_subset = df.loc[df.type == ty, [rate_key, 'preds_dec']]
                ax.plot(df_subset[rate_key], df_subset.preds_dec, label=name)

            ax.set_title(f"Type: {ty}", fontweight='bold')
            ax.set_xlim([min(rates) - 1, max(rates) + 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Rate, Hz', fontweight='bold')
            if ai == 0:
                ax.set_ylabel('Prop fast decision', fontweight='bold')
                ax.legend(title='Set')

        plt.suptitle(self.model.integration_type.capitalize(), fontweight='bold')
        fig.tight_layout()
        plt.show()

    def _plot_for_intermediate_model(self, data: MultiChannel, row: int) -> plt.Figure:

        n_side_plots = len(self.model.left_layers.values()) + len(self.model.left_output_layers.values())
        n_combined_plots = len(self.model.combined_layers.values()) + len(self.model.combined_output_layers.values())
        n_plots = n_side_plots + n_combined_plots

        fig = plt.figure(figsize=(8, n_plots * 3))
        gs = fig.add_gridspec(n_plots, 2)

        # Side stim inputs
        row_idx = 0
        for side_idx, side in enumerate(['left', 'right']):
            ax = fig.add_subplot(gs[row_idx, side_idx])
            ax.plot(data.x_test[f"{side}_x"][row])
            ax.set_title(f"{side.capitalize()} rate: {data.y_test[f'{side}_y_rate'][row]} "
                         f"({data.y_test[f'{side}_y_dec'][row]})")

        # Side intermediate layers
        start_row_idx = row_idx + 1
        for ri, layer_key in enumerate(self.model.left_layers.keys()):
            row_idx = start_row_idx + ri
            for side_idx, side in enumerate(['left', 'right']):
                ax = fig.add_subplot(gs[row_idx, side_idx])
                ax.plot(self.preds_test[f"{side}_{layer_key}"][row], linewidth=0.3)
                ax.set_title(f"{side.capitalize()} {layer_key}")

        # Side output layers
        start_row_idx = row_idx + 1
        for ri, layer_key in enumerate(self.model.left_output_layers.keys()):
            row_idx = start_row_idx + ri
            for side_idx, side in enumerate(['left', 'right']):
                ax = fig.add_subplot(gs[row_idx, side_idx])
                ax.bar(x=[0], height=[float(self.preds_test[f"{side}_{layer_key}"][row])])
                ax.set_ylim([0, 20])
                ax.set_title(f"{side.capitalize()} {layer_key}")

        # Combined layers
        start_row_idx = row_idx + 1
        for ri, layer_key in enumerate(self.model.combined_layers.keys()):
            row_idx = start_row_idx + ri
            ax = fig.add_subplot(gs[row_idx, :])
            ax.plot(self.preds_test[f"{layer_key}"][row], linewidth=0.3)
            ax.set_title(f"Combined {layer_key}")

        # Combined output layers
        row_idx += 1
        ax = fig.add_subplot(gs[row_idx, :])
        ax.bar(['slow', 'fast'], self.preds_test["agg_y_dec"][row])
        ax.set_title(f"Predicted rate: {self.preds_test['agg_y_rate'][row]} "
                     f"({self.preds_test['agg_y_dec'][row]})")

        fig.tight_layout()

        return fig

    def plot_example(self,
                     data: MultiChannel,
                     show: bool = True,
                     dec_key: str = "agg_y_dec",
                     mistake: bool = False):
        """
        Plot a random example from the test set, with output from an early conv layer.

        TODO: Inefficient, predicts for all (to find mistakes).
        TODO: Upgrade to work with subplots for multisensory would be nice.
        """

        self.predict(data)

        if mistake:
            mistakes = ~((self.preds_test[dec_key][:, 1] > 0.5)
                         == (data.y_test[dec_key][:, 1].astype(bool)))
            row = np.random.choice(np.where(mistakes)[0])
        else:
            row = np.random.choice(range(0, self.preds_test[dec_key].shape[0]))

        self._plot_for_intermediate_model(data, row)

        if show:
            plt.show()

    def report(self, data: MultiChannel) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_preds, test_preds = self.predict(data)

        dfs = []
        for summ, d, preds in zip([data.summary_train, data.summary_test],
                                  [data.y_train, data.y_test],
                                  [train_preds, test_preds]):
            report_df = pd.DataFrame({'left_y_rate': d["left_y_rate"],
                                      'right_y_rate': d["right_y_rate"],
                                      'agg_y_rate': d["agg_y_rate"],
                                      'preds_rate': preds["agg_y_rate"].squeeze(),
                                      'left_y_dec': d["left_y_dec"][:, 1],
                                      'right_y_dec': d["right_y_dec"][:, 1],
                                      'agg_y_dec': d["agg_y_dec"][:, 1],
                                      'preds_dec': preds["agg_y_dec"][:, 1]},
                                     index=summ.index)

            merged_df = report_df.merge(summ, left_index=True, right_index=True)
            dfs.append(merged_df)

        return tuple(dfs)
