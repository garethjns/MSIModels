import glob
import os
import tempfile
import unittest

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory


class TestExperimentalRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        limit_gpu_memory(256)

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_data_path = os.path.join(self._tmp_dir.name, "test_data.hdf")
        self._test_run_name = "test_experimental_run"
        self._test_model_name = "test_model"
        # Prepare run
        self._sut = ExperimentalRun(
            name=self._test_run_name,
            model=ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                           opt='adam', batch_size=20, lr=0.01),
                                    name=self._test_model_name),
            data=ExperimentalDataset(self._tmp_dir.name, n=400, difficulty=12).build(self._tmp_data_path),
            n_reps=2, n_epochs=1)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_run_experiment(self):
        self._sut.run()
        self._sut.evaluate()

    def test_save_models_saves_all_reps_to_separate_dirs(self):
        # Act
        self._sut.save_models()

        # Assert
        saved_rep_dirs = glob.glob(os.path.join(self._tmp_dir.name, self._test_model_name, "rep_*"))
        self.assertEqual(2, len(saved_rep_dirs))
