import os
import tempfile
import unittest
from unittest.mock import Mock

from msi_models.experiment.experimental_dataset import ExperimentalDataset


class TestExperimentalDataset(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._sut = ExperimentalDataset(n=10)
        self._test_file_path = os.path.join(self._tmp_dir.name, f"test_file.hdf")

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_build_when_does_not_exist(self):
        # Act
        self._sut.build(self._test_file_path, n_jobs=1)

        # Assert
        self.assertTrue(os.path.exists(self._test_file_path))

    def test_build_when_exists(self):
        # Arrange
        with open(self._test_file_path, 'w') as f:
            f.write('test file')
        self._sut._load = Mock()

        # Act
        self._sut.build(self._test_file_path, n_jobs=1)

        # Assert
        self._sut._load.assert_called_once()

    def _assert_output_lengths(self, n=10, n_train=8, n_test=2):
        self.assertEqual(n, self._sut.mc.summary.shape[0])
        self.assertEqual(n_train, self._sut.mc.summary_train.shape[0])
        self.assertEqual(n_test, self._sut.mc.summary_test.shape[0])
        self.assertEqual(n, self._sut.mc.x['left_x'].shape[0])
        self.assertEqual(n_train, self._sut.mc.x_train['left_x'].shape[0])
        self.assertEqual(n_test, self._sut.mc.x_test['left_x'].shape[0])
        self.assertEqual(n, self._sut.mc.x['right_x'].shape[0])
        self.assertEqual(n_train, self._sut.mc.x_train['right_x'].shape[0])
        self.assertEqual(n_test, self._sut.mc.x_test['right_x'].shape[0])
        self.assertEqual(n, len(self._sut.mc.y['agg_y_rate']))
        self.assertEqual(n_train, len(self._sut.mc.y_train['agg_y_rate']))
        self.assertEqual(n_test, len(self._sut.mc.y_test['agg_y_rate']))

    def test_train_test_split_with_defaults(self):
        # Act
        self._sut.build(self._test_file_path, n_jobs=1)

        # Assert
        self._assert_output_lengths(n=10, n_train=8, n_test=2)

    def test_test_train_split_when_prop_specified(self):
        # Arrange
        new_prop = 0.6
        self._sut.common_channel_kwargs.update({'train_prop': new_prop})

        # Act
        self._sut.build(self._test_file_path, n_jobs=1, batch_size=1)

        # Assert
        self.assertEqual(new_prop, self._sut.common_channel_kwargs['train_prop'])
        self.assertEqual(new_prop, self._sut.mc.config.channels[0].train_prop)
        self.assertEqual(new_prop, self._sut.mc.config.channels[1].train_prop)
        self._assert_output_lengths(n=10, n_train=6, n_test=4)
