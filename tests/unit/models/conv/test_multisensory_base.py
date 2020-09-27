import unittest

from msi_models.models.conv.multisensory_base import MultisensoryBase


class TestMultisensoryBase(unittest.TestCase):
    _sut = MultisensoryBase

    def test_early_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='early_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        self.assertAlmostEqual(160000, mod.n_params, -4)

    def test_intermediate_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='intermediate_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        self.assertAlmostEqual(160000, mod.n_params, -4)

    def test_late_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='late_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        self.assertAlmostEqual(160000, mod.n_params, -4)
