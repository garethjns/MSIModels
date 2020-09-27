import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fit_psyche.psychometric_curve import PsychometricCurve

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate


class ExperimentalResults:
    """Handles results, evaluation, and plotting, for an ExperimentalRun (repeats of a model/data combination)."""

    def __init__(self, rate_key: str = 'agg_y_rate', type_key: str = 'type') -> None:
        self.rate_key: str = rate_key
        self.type_key: str = type_key

        # Set with .add_models() model is added (repeats of same model)
        self.mods: Union[None, List[ExperimentalModel]] = None

        # Set with .add_data(). Single dataset is used for each Run, but training tr/val split will have been different
        # for each.
        self.types: List[int] = []
        self.rates: List[int] = []
        self.subjects: List[int] = []
        self.data: Union[None, ExperimentalDataset] = None

        # Set on evaluation
        self.model_perf_subject: Union[None, pd.DataFrame] = None
        self.model_perf_agg: Union[None, pd.DataFrame] = None
        self.pfs_subject: Union[None, pd.DataFrame] = None
        self.pfs_agg: Union[None, pd.DataFrame] = None
        self.curves_subject: Union[None, pd.DataFrame] = None
        self.curves_agg: Union[None, pd.DataFrame] = None

    def set_data(self, data: ExperimentalDataset) -> "ExperimentalResults":
        if self.data is None:
            self.data = data
            self.types = self.data.types
            self.rates = self.data.rates
        else:
            raise RuntimeError(f"Data already set, create a new object to set new.")

        return self

    def add_models(self, models: List[ExperimentalModel]) -> "ExperimentalResults":
        if self.mods is None:
            self.mods = models
            self.subjects = list(range(len(models)))
        else:
            raise RuntimeError(f"Mods already set, create a new object to set new.")

        return self

    def evaluate_subjects(self) -> None:
        """
        Eval model performance, calculate and aggregate prop fasts, and fit curves over subject.

        Done by model to avoid time/memory cost of re-predicting and/or keeping predictions in memory.
        """

        model_perfs = []
        pfs = []
        curves = []
        for si, exp_mod in zip(self.subjects, self.mods):
            # Basic model performance (overall, TODO: By type).
            mp = exp_mod.evaluate_performance(self.data)
            mp.loc[:, 'subject'] = si
            model_perfs.append(mp)

            # Prop fasts
            pf = exp_mod.calc_prop_fasts(self.data, type_key=self.type_key, rate_key=self.rate_key)
            pf.loc[:, 'subject'] = si
            pfs.append(pf)

            # Psyche curve fits
            cur = exp_mod.calc_psyche_curves(self.data, type_key=self.type_key, rate_key=self.rate_key)
            cur.loc[:, 'subject'] = si
            curves.append(cur)

            exp_mod.clear()

        self.model_perf_subject = pd.concat(model_perfs, axis=0)
        self.pfs_subject = pd.concat(pfs, axis=0).reset_index(drop=True)
        self.curves_subject = pd.concat(curves, axis=0).reset_index(drop=True)

    def _aggregate_model_perf_over_subject(self) -> None:
        cols = [c for c in self.model_perf_subject.columns if c != 'subject']
        gb = self.model_perf_subject[cols].groupby('set')

        self.model_perf_agg = pd.concat((gb.mean().rename({c: f"{c}_mean" for c in cols}, axis=1),
                                         gb.std().rename({c: f"{c}_std" for c in cols}, axis=1),
                                         gb.count().rename({c: f"{c}_n" for c in cols}, axis=1)), axis=1)

    def _aggregate_pfs_over_subjects(self) -> None:
        gb = self.pfs_subject[[c for c in self.pfs_subject if c != 'subject']].groupby(
            ['set', self.type_key, self.rate_key])

        self.pfs_agg = pd.concat((gb.mean().rename({'preds_dec': 'preds_dec_mean'}, axis=1),
                                  gb.std().rename({'preds_dec': 'preds_dec_std'}, axis=1),
                                  gb.count().rename({'preds_dec': 'preds_dec_n'}, axis=1)), axis=1)

    def _aggregate_curves_over_subject(self) -> None:
        gb = self.curves_subject[[c for c in self.curves_subject.columns
                                  if c not in ['subject', 'model']]].groupby(['type', 'set'])

        cols = ['mean', 'var', 'guess_rate', 'lapse_rate']
        self.curves_agg = pd.concat((gb.mean().rename({k: f"{k}_mean" for k in cols}, axis=1),
                                     gb.std().rename({k: f"{k}_std" for k in cols}, axis=1),
                                     gb.count().rename({'mean': 'n'}, axis=1)['n']), axis=1)

    def aggregate_over_subjects(self) -> None:
        """
        For prop_fasts: Average over subject, keep train/test, type, rate
        For curves: Average parameters over subjects
        """
        if (self.pfs_subject is None) or (self.curves_subject is None):
            self.evaluate_subjects()

        self._aggregate_model_perf_over_subject()
        self._aggregate_pfs_over_subjects()
        self._aggregate_curves_over_subject()

    def evaluate(self):
        self.evaluate_subjects()
        self.aggregate_over_subjects()

    @staticmethod
    def _subset_to_plot(train: bool = True, test: bool = True) -> List[str]:
        sets = []
        if train:
            sets += ['train']
        if test:
            sets += ['test']

        return sets

    def plot_aggregated_results(self, train: bool = False, test: bool = True,
                                include_curves: bool = True, show: bool=True, path: str = None) -> plt.Figure:
        n_subplots = len(self.types)
        fig, axs = plt.subplots(ncols=n_subplots, figsize=(2.5 * n_subplots, 8))

        sets = self._subset_to_plot(train, test)
        data = self.pfs_agg.reset_index(drop=False)
        curve_data = self.curves_agg.reset_index(drop=False)

        for ai, (ax, ty) in enumerate(zip(axs, self.types)):
            for name in sets:
                idx = (data[self.type_key] == ty) & (data["set"] == name)
                data_subset = data.loc[idx, :].sort_values(self.rate_key, ascending=True)

                ax.scatter(data_subset[self.rate_key], data_subset['preds_dec_mean'], label=name)
                ax.errorbar(data_subset[self.rate_key], data_subset['preds_dec_mean'],
                            data_subset['preds_dec_std'] / np.sqrt(data_subset['preds_dec_n']), linestyle=' ')

                if include_curves:
                    curve_idx = (curve_data[self.type_key] == ty) & (curve_data['set'] == name)
                    pc = PsychometricCurve(model='wh')
                    pc.coefs_ = {k: curve_data.loc[curve_idx, f"{k}_mean"].values[0]
                                 for k in ['mean', 'var', 'guess_rate', 'lapse_rate']}
                    x = np.linspace(min(self.rates), max(self.rates), 200)
                    y_pred = pc.predict(x)
                    ax.plot(x, y_pred, label=f"{name}_fit \n(b={np.round(pc.coefs_['mean'], 2)}\n"
                                             f"var={np.round(pc.coefs_['var'], 2)}\n"
                                             f"gr={np.round(pc.coefs_['guess_rate'], 2)}\n"
                                             f"lr={np.round(pc.coefs_['lapse_rate'], 2)})")

            ax.set_ylim([0, 1])
            ax.set_xlim([min(self.rates) - 2, max(self.rates) + 2])
            ax.set_xlabel('Rate, Hz', fontweight='bold')
            ax.set_title(str(MultiTwoGapTemplate(ty)).split('.')[1])
            ax.legend()
            if ai == 0:
                ax.set_ylabel('Prop fast decision', fontweight='bold')

        fig.suptitle(f"{self.mods[0].model.integration_type.capitalize()} on {self.data.name} (n={len(self.mods)})",
                     fontweight='bold')

        if path is not None:
            fig.savefig(os.path.join(path, "aggregated_results.png"))

        if show:
            fig.show()

        return fig

    def save(self, path: str) -> None:
        for name in ["model_perf_subject", "model_perf_agg", "pfs_subject", "pfs_agg", "curves_subject", "curves_agg"]:
            getattr(self, name).to_csv(f"{os.path.join(path, name)}.csv")
