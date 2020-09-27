import os
import tempfile
import unittest
from unittest.mock import Mock

from msi_models.experiment.experimental_dataset import ExperimentalDataset


class TestExperimentalDataset(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._sut = ExperimentalDataset(n=10)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_build_when_does_not_exist(self):
        # Arrange
        test_file_path = os.path.join(self._tmp_dir.name, f"test_1.hdf")

        # Act
        self._sut.build(test_file_path)

        # Assert
        self.assertTrue(os.path.exists(test_file_path))

    def test_build_when_exists(self):
        # Arrange
        test_file_path = os.path.join(self._tmp_dir.name, f"test_2.hdf")
        with open(test_file_path, 'w') as f:
            f.write('test file')
        self._sut._load = Mock()

        # Act
        self._sut.build(test_file_path)

        # Assert
        self._sut._load.assert_called_once()
