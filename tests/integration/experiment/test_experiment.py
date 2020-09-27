import os
import tempfile
import unittest

from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        limit_gpu_memory(256)

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_data_path_1 = os.path.join(self._tmp_dir.name, "test_data.hdf")
        self._tmp_data_path_2 = os.path.join(self._tmp_dir.name, "test_data.hdf")

        self._sut = Experiment(name='test_experiment', n_epochs=2, n_reps=3)

        datasets = [ExperimentalDataset("easy", n=256, difficulty=12).build(self._tmp_data_path_1),
                    ExperimentalDataset("hard", n=256, difficulty=12).build(self._tmp_data_path_2)]

        for data in datasets:
            self._sut.add_data(data)

        common_model_kwargs = {'opt': 'adam', 'batch_size': 50, 'lr': 0.01}
        for int_type in ['early_integration', 'intermediate_integration', 'late_integration']:
            mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs),
                                    name=int_type)
            self._sut.add_model(mod)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_run_experiment(self):
        self._sut.run()
