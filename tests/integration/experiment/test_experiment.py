import os
import tempfile
import unittest

import tensorflow as tf

from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate

try:
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=256)])
except Exception:
    pass


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_data = os.path.join(self._tmp_dir.name, "test_data.hdf")

        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['left_only'], MultiTwoGapTemplate['right_only'],
                                            MultiTwoGapTemplate['matched_sync'], MultiTwoGapTemplate['matched_async'],
                                            MultiTwoGapTemplate['unmatched_async']],
                                 fs=500, n=200, batch_size=2, fn=self._tmp_data, n_jobs=5,
                                 template_kwargs={"duration": 1300, "background_mag": 0.3, "duration_tol": 0.5})

        self._sut = Experiment(name='example_experiment', n_epochs=2, n_reps=3)
        self._sut.add_data(self._tmp_data)
        common_model_kwargs = {'opt': 'adam', 'batch_size': 50, 'lr': 0.01}
        for int_type in ['early_integration', 'intermediate_integration', 'late_integration']:
            mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs),
                                    name=int_type)
            self._sut.add_model(mod)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_run_experiment(self):
        self._sut.run()

