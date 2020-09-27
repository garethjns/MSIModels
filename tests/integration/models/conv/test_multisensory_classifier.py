import tempfile
import unittest

from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory


class TestMultisensoryClassifier(unittest.TestCase):
    def setUp(self):
        limit_gpu_memory(512)
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._sut = MultisensoryClassifier()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_save_load_unbuilt_model(self):
        # Act
        self._sut.save(self._tmp_dir.name)
        new_obj = self._sut.load(self._tmp_dir.name)

        # Assert
        self.assertIsInstance(new_obj, MultisensoryClassifier)
        self.assertIsNone(new_obj.model)

    def test_save_load_build_model(self):
        # Arrange
        self._sut.build_model()

        # Act
        self._sut.save(self._tmp_dir.name)
        new_obj = self._sut.load(self._tmp_dir.name)

        # Assert
        self.assertIsInstance(new_obj, MultisensoryClassifier)
        self.assertIsNotNone(new_obj.model)

    def test_plot_dag(self):
        # Arrange
        self._sut.build_model()

        # Act
        self._sut.plot_dag(path=self._tmp_dir.name)

        # No assert on file exists as plot allowed to fail when graphviz/pydot not available.
