import os
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from msi_models.experiment.experimental_run import ExperimentalRun


class ExperimentalResults:
    _model_axis_label = "Model name"
    _data_axis_label = "Dataset name"

    def __init__(self, experiment_path: str) -> None:
        self.experiment_path = experiment_path
        self.output_path = experiment_path
        self.graphs_path = os.path.join(self.experiment_path, "graphs")
        pathlib.Path(self.graphs_path).mkdir(exist_ok=True)

        self.results_subjects_csvs: Dict[str, str] = {}
        self.results_subjects: pd.DataFrame = pd.DataFrame()

    def plot_all(self) -> None:
        for subset in ['all', 'train', 'test']:
            self.plot_dt(subset)
            self.plot_dt_by_type(subset)
            self.plot_model_perf(subset)

    def evaluate(self) -> None:
        """High level analysis across models and datasets."""
        for name, df in self.results_subjects.items():
            self.results_subjects_csvs[str(name)] = os.path.join(self.output_path, f"results_{name}.csv")
            df.to_csv(self.results_subjects_csvs[str(name)])

    def add_results(self, runs: List[ExperimentalRun]) -> None:
        collect = ["curves_subject", "model_perf_subject", "pfs_subject"]
        dfs = {k: [] for k in collect}
        for run in runs:
            for results_set_name in ["curves_subject", "model_perf_subject", "pfs_subject"]:
                data = getattr(run.results, results_set_name)
                data.loc[:, "model_name"] = run.model.name
                data.loc[:, "dataset_name"] = run.data.name
                dfs[results_set_name].append(data)

        self.results_subjects = {k: pd.concat(v, axis=0) for k, v in dfs.items()}

    def plot_dt_by_type(self, subset: str = 'test', show: bool = True) -> List[plt.Figure]:
        data = self.results_subjects["curves_subject"].rename({'model_name': self._model_axis_label,
                                                               'dataset_name': self._data_axis_label}, axis=1)
        if subset != "all":
            data = data.loc[data.set == subset]

        figs = []
        for ty in data.type.unique():
            fig, ax = plt.subplots()
            ax = sns.boxplot(data=data.loc[data.type == ty, :], x=self._model_axis_label, y='var',
                             hue=self._data_axis_label, ax=ax)
            ax.set_title(f'Subset: {subset}, type: {ty}', fontweight='bold')
            ax.set_ylabel('Discrimination threshold', fontweight='bold')
            ax.get_legend().get_title().set_fontweight('bold')
            ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            fig.savefig(os.path.join(self.graphs_path, f"dt_{subset}_type{ty}.png"))
            figs.append(fig)

            if show:
                plt.show()

        return figs

    def plot_dt(self, subset: str = 'test', show: bool = True) -> plt.Figure:
        data = self.results_subjects["curves_subject"].rename({'model_name': self._model_axis_label,
                                                               'dataset_name': self._data_axis_label}, axis=1)
        if subset != "all":
            data = data.loc[data.set == subset]

        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=self._model_axis_label, y='var', hue=self._data_axis_label, ax=ax)
        ax.set_title(f'Subset: {subset}, type: all', fontweight='bold')
        ax.set_ylabel('Discrimination threshold', fontweight='bold')
        ax.get_legend().get_title().set_fontweight('bold')
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
        fig.savefig(os.path.join(self.graphs_path, f"dt_{subset}.png"))

        if show:
            plt.show()

        return fig

    def plot_model_perf(self, subset: str = 'test', show: bool = True) -> plt.Figure:
        data = self.results_subjects["model_perf_subject"].rename({'model_name': self._model_axis_label,
                                                                   'dataset_name': self._data_axis_label}, axis=1)
        if subset != "all":
            data = data.loc[data.set == subset]

        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=self._model_axis_label, y='dec_accuracy', hue=self._data_axis_label)
        ax.set_title(f'Subset: {subset}, type: all', fontweight='bold')
        ax.set_ylabel('Decision accuracy', fontweight='bold')
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
        fig.savefig(os.path.join(self.graphs_path, f"dt_{subset}.png"))

        if show:
            plt.show()

        return fig
