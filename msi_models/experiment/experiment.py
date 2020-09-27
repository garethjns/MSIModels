import os
import pathlib
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_results import ExperimentalResults
from msi_models.experiment.experimental_run import ExperimentalRun


class Experiment:
    def __init__(self, name: str = 'unnamed_experiment', path: str = '', n_reps: int = 5, n_epochs: int = 2000) -> None:
        self.name = name
        self._path = path
        self.n_reps = n_reps
        self.n_epochs = n_epochs

        self.models: List[ExperimentalModel] = []
        self.datasets: List[ExperimentalDataset] = []

        self._runs: List[ExperimentalRun] = []

        self.output_path, self._i = self._get_next_output_path()
        self.results = ExperimentalResults(experiment_path=self.output_path)

    def add_model(self, mod: ExperimentalModel) -> None:
        if mod not in self.models:
            self.models.append(mod)
            mod.model.plot_dag(path=self.output_path)

    def add_data(self, data: ExperimentalDataset) -> None:
        if data not in self.datasets:
            self.datasets.append(data)
            data.mc.plot_summary(subset='train', show=False).savefig(os.path.join(self.output_path,
                                                                                  f"{data.name}_train.png"))
            data.mc.plot_summary(subset='test', show=False).savefig(os.path.join(self.output_path,
                                                                                 f"{data.name}_test.png"))

    def _gen_path(self, i: int) -> str:
        return os.path.join(self._path, self.name, f'_{i}')

    def _get_next_output_path(self) -> Tuple[str, int]:
        i = 0
        path = self._gen_path(i)

        while os.path.exists(path):
            i += 1
            path = self._gen_path(i)

        pathlib.Path(path).mkdir(exist_ok=False, parents=True)

        return path, i

    def _generate_runs(self) -> None:
        self._runs = []
        for mod, data in np.array(np.meshgrid(self.models, self.datasets)).T.reshape(-1, 2):
            self._runs.append(ExperimentalRun(name=f"{self.name}_{mod.name}_on_{data.name}", model=mod, data=data,
                                              n_reps=self.n_reps, n_epochs=self.n_epochs, exp_path=self.output_path))

    def run(self) -> None:
        self._generate_runs()

        for exp_run in tqdm(self._runs, desc=self.name):
            exp_run.run()
            exp_run.evaluate()
            exp_run.plot()
            exp_run.log_run(to=f"{self.name}")
            exp_run.log_summary(to=f"{self.name}_summary")
            exp_run.save_models()

        self.evaluate()

    def evaluate(self) -> None:
        self.results.add_results(self._runs)
        self.results.evaluate()
        self.results.plot_all()
