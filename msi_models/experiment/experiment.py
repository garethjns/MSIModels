from typing import List, Union

from tqdm import tqdm

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel
from msi_models.stimset.multi_channel_config import MultiChannelConfig


class Experiment:
    def __init__(self, name: str = 'unnamed_experiment', n_reps: int = 5, n_epochs: int = 2000) -> None:
        self.name = name
        self.n_reps = n_reps
        self.n_epochs = n_epochs

        self.models: List[ExperimentalModel] = []
        self.dataset: Union[None, MultiChannel] = None
        self._runs: List[ExperimentalRun] = []

    def add_model(self, mod: ExperimentalModel) -> None:
        if mod not in self.models:
            self.models.append(mod)

    def add_data(self, path: str = "data/sample_multisensory_data_mix_hard_250k.hdf5") -> None:
        if self.dataset is not None:
            raise RuntimeError(f"Data already added.")
        else:
            common_channel_kwargs = {"path": path, "train_prop": 0.8, "x_keys": ["x", "x_mask"],
                                     "y_keys": ["y_rate", "y_dec"]}

            multi_config = MultiChannelConfig(path=path, key='agg', y_keys=common_channel_kwargs["y_keys"],
                                              channels=[ChannelConfig(key='left', **common_channel_kwargs),
                                                        ChannelConfig(key='right', **common_channel_kwargs)])
            data = MultiChannel(multi_config)

            self.dataset = data

    def _generate_runs(self) -> None:
        self._runs = []
        for mod in self.models:
            self._runs.append(ExperimentalRun(name=f"{self.name}_{mod.name}", model=mod, data=self.dataset,
                                              n_reps=self.n_reps, n_epochs=self.n_epochs))

    def run(self) -> None:
        self._generate_runs()

        for exp_run in tqdm(self._runs, desc=self.name):
            exp_run.run()
            exp_run.evaluate()
            exp_run.plot()
            exp_run.log_run(to=f"{self.name}")
            exp_run.log_summary(to=f"{self.name}_summary")
