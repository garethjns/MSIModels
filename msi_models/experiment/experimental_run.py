import copy
from dataclasses import dataclass
from typing import Dict, List, Any

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel


@dataclass
class ExperimentalRun:
    data: ExperimentalDataset
    model: ExperimentalModel
    name: str = "unnamed_experiment_run"
    n_reps: int = 5
    validation_split: float = 0.4

    path: str = None  # TODO: Output path, but not saving anything yet.
    preds_y_train: Dict[str, np.ndarray] = None
    preds_y_test: Dict[str, np.ndarray] = None
    run_results: List[Dict[str, float]] = None

    _mlflow_run_ids: List[int] = None
    _mlflow_exp_id: str = None
    _models: List[ExperimentalModel] = None
    _data_samples: List[ExperimentalDataset] = None

    def __post_init__(self):
        self._mlflow_exp_id = mlflow.set_experiment(self.name)
        self._mlflow_run_ids = []
        self._models = []
        self._data_samples = []
        self.run_results = []

        self._prepare_runs()

    def _prepare_runs(self):
        for r in range(self.n_reps):
            self._models.append(copy.deepcopy(self.model))
            data_sample = copy.deepcopy(self.data)
            data_sample.build(seed=r)
            self._data_samples.append(data_sample)

    def run(self):
        for r in range(self.n_reps):
            self._mlflow_run_ids.append(mlflow.start_run(experiment_id=self._mlflow_exp_id))

            self._fit(self._models[r], self._data_samples[r])
            self.run_results.append(self._evaluate(self._models[r], self._data_samples[r]))

            mlflow.log_param('rep', r)
            self.log_common()
            self._log_results(self.run_results[r])
            mlflow.end_run()

            tf.keras.backend.clear_session()

    def _fit(self, model, data, **kwargs):
        model.fit(data,
                  validation_split=self.validation_split,
                  **kwargs)

    @staticmethod
    def _evaluate(model, data) -> Dict[str, float]:
        preds_y_train, preds_y_test = model.predict(data)

        return {'train_rate_mse': mean_squared_error(data.stimset.y_train['agg_y_rate'],
                                                     preds_y_train["agg_y_rate"]),
                'test_rate_mse': mean_squared_error(data.stimset.y_test['agg_y_rate'],
                                                    preds_y_test["agg_y_rate"]),
                'train_dec_accuracy': accuracy_score(data.stimset.y_train['agg_y_dec'][:, 1],
                                                     preds_y_train["agg_y_dec"][:, 1] > 0.5),
                'test_dec_accuracy': accuracy_score(data.stimset.y_test['agg_y_dec'][:, 1],
                                                    preds_y_test["agg_y_dec"][:, 1] > 0.5)}

    def log_common(self):
        mlflow.log_param('dataset_name', self.data.name)
        mlflow.log_param('dataset_path', self.data.config.path)
        mlflow.log_param('model_name', self.model.name)

    @staticmethod
    def _log_results(results: Dict[str, Any]):
        """Log to mlflow."""
        mlflow.log_metrics(results)

    @property
    def results(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame(r, index=[ri]) for ri, r in enumerate(self.run_results)],
                         axis=0)

    @property
    def agg_results(self) -> pd.Series:
        return self.results.describe()
