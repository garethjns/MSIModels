import unittest

from msi_models.models.conv.multisensory_base import MultisensoryBase


class TestMultisensoryBase(unittest.TestCase):
    """
    TODO: Add asserts for param counts when fixed: np.sum([count_params(w) for w in mod.model.trainable_weights])
    """
    _sut = MultisensoryBase

    def test_early_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='early_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)

    def test_intermediate_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='intermediate_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)

    def test_late_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='late_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
