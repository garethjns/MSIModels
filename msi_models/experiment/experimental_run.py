import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlflow
import tensorflow as tf
from tqdm import tqdm

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_results import ExperimentalResults
from msi_models.stimset.multi_channel import MultiChannel


@dataclass
class ExperimentalRun:
    """Handles repetition and evaluation for a data and model combination."""
    data: MultiChannel
    model: ExperimentalModel
    name: str = "unnamed_experiment_run"
    n_reps: int = 5
    n_epochs: int = 2000
    validation_split: float = 0.4

    def __post_init__(self) -> None:
        self.run_results = []
        self.done: bool = False
        self.results = ExperimentalResults()

        self._mlflow_run_ids: Dict[Tuple[int, int], int] = {}
        self._models: List[ExperimentalModel] = []

        self._prepare_runs()

    def _prepare_runs(self) -> None:
        for r in range(self.n_reps):
            self._models.append(copy.deepcopy(self.model))

    def run(self) -> None:
        if self.done:
            warnings.warn('Already run.')

        if not self.done:
            self.results.set_data(self.data)

            for r in tqdm(range(self.n_reps), desc=self.name):
                self._fit(self._models[r], self.data, epochs=self.n_epochs)
                tf.keras.backend.clear_session()

            self.results.add_models(self._models)
            self.done = True

    def evaluate(self) -> None:
        self.results.evaluate()

    def plot(self) -> None:
        self.results.plot_aggregated_results()

    def clear(self) -> None:
        for mod in self._models:
            mod.clear()

    def _fit(self, model, data, **kwargs) -> None:
        model.fit(data, validation_split=self.validation_split, **kwargs)

    def _log_common_params(self):
        mlflow.log_param('dataset_path', self.data.config.path)
        mlflow.log_param('integration_type', self.model.name.split('_')[0])

    def log_run(self, to: str) -> None:
        """ Log each type of each repeat/"subject" as a run."""

        mlflow.set_experiment(to)

        for si, mod in enumerate(self._models):
            for ty in self.results.types:
                self._mlflow_run_ids[(si, ty)] = mlflow.start_run(run_name=f"Subject: {si}, type: {ty}")
                self._log_common_params()
                mlflow.log_metrics({'type': ty, 'subject': si})

                # Curves
                idx = (self.results.curves_subject.subject == si) & (self.results.curves_subject.type == ty)
                results = self.results.curves_subject.loc[idx, :]
                mlflow.log_metrics({'bias_train': results.loc[results['set'] == 'train', 'mean'].values[0],
                                    'bias_test': results.loc[results['set'] == 'test', 'mean'].values[0],
                                    'dt_train': results.loc[results['set'] == 'train', 'var'].values[0],
                                    'dt_test': results.loc[results['set'] == 'test', 'var'].values[0]})

                mlflow.end_run()

    def log_summary(self, to: str) -> None:
        """Log a single summary "run" to another experiment (such as parent Experiment)."""
        mlflow.set_experiment(to)
        mlflow.start_run(run_name=self.model.name)
        self._log_common_params()

        for dset in ["train", "test"]:
            idx = self.results.model_perf_agg.index == dset
            to_log = ['dec_accuracy_mean', 'rate_loss_mean', 'dec_accuracy_std', 'rate_loss_std']
            mlflow.log_metrics({k: self.results.model_perf_agg.loc[idx, k].values[0] for k in to_log})

            curves_df = self.results.curves_agg.reset_index(drop=False)
            to_log = ['bias_mean', 'dt_mean', 'bias_std', 'dt_std']
            for ty in self.results.types:
                idx = (curves_df["set"] == dset) & (curves_df["type"] == ty)
                mlflow.log_metrics({f"{k}_{dset}": curves_df.loc[idx, k].values[0] for k in to_log})

        mlflow.end_run()
