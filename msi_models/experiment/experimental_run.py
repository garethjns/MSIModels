import copy
import numbers
import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlflow
import tensorflow as tf
from tqdm import tqdm

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run_results import ExperimentalResults


@dataclass
class ExperimentalRun:
    """Handles repetition and evaluation for a data and model combination."""
    data: ExperimentalDataset
    model: ExperimentalModel
    name: str = "unnamed_experiment_run"
    exp_path: str = "unnamed_experiment/"
    n_reps: int = 5
    n_epochs: int = 2000
    validation_split: float = 0.4

    def __post_init__(self) -> None:
        self.run_results = []
        self.done: bool = False
        self.results = ExperimentalResults()

        self._mlflow_run_ids: Dict[Tuple[int, int], int] = {}
        self._models: List[ExperimentalModel] = []

        self._prepare_model()
        self._prepare_output_path()
        self._prepare_runs()

        # TODO: Temp numeric id for integration types - replace with Enum
        self._model_type_map: Dict[str, int] = {'early_integration': 0, 'intermediate_integration': 1,
                                                'late_integration': 2}

    def _prepare_model(self):
        self.model.model.clear_tf()

    def _prepare_output_path(self) -> None:
        self.output_path = os.path.join(self.exp_path, self.data.name, self.model.name)
        pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def _prepare_runs(self) -> None:
        for _ in range(self.n_reps):
            self._models.append(copy.deepcopy(self.model))

    def run(self) -> None:
        if self.done:
            warnings.warn('Already run.')

        if not self.done:
            self.results.set_data(self.data)

            for r in tqdm(range(self.n_reps), desc=self.name):
                self._fit(self._models[r], self.data, epochs=self.n_epochs, verbose=1)
                tf.keras.backend.clear_session()

            self.results.add_models(self._models)
            self.done = True

    def evaluate(self) -> None:
        self.results.evaluate()
        self.results.save(self.output_path)

    def plot(self) -> None:
        self.results.plot_aggregated_results(path=self.output_path)

    def clear(self) -> None:
        for mod in self._models:
            mod.clear()

    def _fit(self, model, data, **kwargs) -> None:
        model.fit(data, validation_split=self.validation_split, **kwargs)

    def _log_common_params(self) -> None:
        mlflow.log_param('dataset_path', self.data.path)
        mlflow.log_param('dataset_name', self.data.name)
        mlflow.log_param('dataset_difficulty', self.data.difficulty)
        mlflow.log_param('dataset_n', self.data.n)
        mlflow.log_param('dataset_n_train', self.data.mc.summary_train.shape[0])
        mlflow.log_param('dataset_n_test', self.data.mc.summary_test.shape[0])
        mlflow.log_param('model_integration_type', self.model.name.split('_')[0])
        mlflow.log_param('model_integration_type_num', self._model_type_map.get(self.model.model.integration_type, -1))
        mlflow.log_param('model_n_params', self._models[0].model.n_params)  # (Needs to be a built model)
        mlflow.log_param('exp_path', self.exp_path)
        mlflow.log_param('output_path', self.output_path)

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
                mlflow.log_metrics({
                    'pc_bias_train': results.loc[results['set'] == 'train', 'mean'].values[0],
                    'pc_bias_test': results.loc[results['set'] == 'test', 'mean'].values[0],
                    'pc_var_train': results.loc[results['set'] == 'train', 'var'].values[0],
                    'pc_var_test': results.loc[results['set'] == 'test', 'var'].values[0],
                    'pc_guess_rate_train': results.loc[results['set'] == 'train', 'guess_rate'].values[0],
                    'pc_guess_rate_test': results.loc[results['set'] == 'test', 'guess_rate'].values[0],
                    'pc_lapse_rate_train': results.loc[results['set'] == 'train', 'lapse_rate'].values[0],
                    'pc_lapse_rate_test': results.loc[results['set'] == 'test', 'lapse_rate'].values[0]})

                mlflow.end_run()

    def log_summary(self, to: str) -> None:
        """Log a single summary "run" to another experiment (such as parent Experiment)."""
        mlflow.set_experiment(to)
        run_id = mlflow.start_run(run_name=self.model.name)
        self._log_common_params()

        for dset in ["train", "test"]:
            idx = self.results.model_perf_agg.index == dset
            to_log = ['dec_accuracy_mean', 'rate_loss_mean', 'dec_accuracy_std', 'rate_loss_std']
            to_log_dict = {}
            for k in to_log:
                v = self.results.model_perf_agg.loc[idx, k].values[0]
                to_log_dict[f"{k}_{dset}"] = v if isinstance(v, numbers.Number) else -1.0
            mlflow.log_metrics(to_log_dict)

            curves_df = self.results.curves_agg.reset_index(drop=False)
            # From : to
            to_log = ['mean_mean', 'var_mean', 'guess_rate_mean', 'lapse_rate_mean',
                      'mean_std', 'var_std', 'guess_rate_std', 'lapse_rate_std']
            for ty in self.results.types:
                idx = (curves_df["set"] == dset) & (curves_df["type"] == ty)
                mlflow.log_metrics({f"pc_{k}_{dset}": curves_df.loc[idx, k].values[0] for k in to_log})

        mlflow.log_artifacts(self.output_path)
        mlflow.end_run()

    def save_models(self) -> None:
        for rep, mod in enumerate(self._models):
            mod.save(os.path.join(self.output_path, f"rep_{rep}"))
