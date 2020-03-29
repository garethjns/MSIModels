from dataclasses import dataclass
from typing import List, Dict

import mlflow
import numpy as np

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun


@dataclass
class Experiment:
    datasets: List[ExperimentalDataset]
    models: List[ExperimentalModel]
    name: str = 'unnamed_experiment'

    _runs: List[ExperimentalRun] = None

    def __post_init__(self):
        self._runs = []
        self._generate_grid()

    def _generate_grid(self):
        grid = np.array(np.meshgrid(self.datasets, self.models)).T.reshape(-1, 2)
        for d, m in grid:
            self._runs.append(ExperimentalRun(model=m, data=d,
                                              name=f"{self.name}_reps"))

    def run(self):
        for exp_run in self._runs:
            exp_run.run()

            mlflow.set_experiment(self.name)
            mlflow.start_run()
            exp_run.log_common()
            self._log_results(exp_run.agg_results.to_dict())
            mlflow.end_run()

    def _log_results(self, results: Dict[str, Dict[str, float]]):
        for col_name, col_vals in results.items():
            col_vals_ = {f"{col_name}_{''.join([s for s in k if s.isalpha()])}": v for k, v in col_vals.items()}

            mlflow.log_metrics(col_vals_)
