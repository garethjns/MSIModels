import os
import tempfile
import unittest

import tensorflow as tf

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

try:
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=256)])
except Exception:
    pass


class TestExperimentalRun(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_data = os.path.join(self._tmp_dir.name, "test_data.hdf")

        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['left_only'], MultiTwoGapTemplate['right_only'],
                                            MultiTwoGapTemplate['matched_sync'], MultiTwoGapTemplate['matched_async'],
                                            MultiTwoGapTemplate['unmatched_async']],
                                 fs=500, n=50, batch_size=5, fn=self._tmp_data, n_jobs=5,
                                 template_kwargs={"duration": 1300, "background_mag": 0.3, "duration_tol": 0.5})

        common_kwargs = {"path": self._tmp_data, "train_prop": 0.8, "seed": 100,
                         "x_keys": ["x", "x_mask"], "y_keys": ["y_rate", "y_dec"]}
        multi_config = MultiChannelConfig(path=self._tmp_data, key='agg', y_keys=common_kwargs["y_keys"],
                                          channels=[ChannelConfig(key='left', **common_kwargs),
                                                    ChannelConfig(key='right', **common_kwargs)])

        # Prepare run
        self._sut = ExperimentalRun(
            name=f"example_run_for_example_model",
            model=ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                           opt='adam', batch_size=20, lr=0.01),
                                    name='example_model'),
            data=MultiChannel(multi_config), n_reps=2, n_epochs=2)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_run_experiment(self):
        self._sut.run()
        self._sut.evaluate()
