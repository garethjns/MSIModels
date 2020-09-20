import unittest

from msi_models.models.conv.multisensory_base import MultisensoryBase
from tensorflow.keras.backend import count_params
import numpy as np


class TestMultisensoryBase(unittest.TestCase):
    _sut = MultisensoryBase

    def test_early_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='early_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        param_count = np.sum([count_params(w) for w in mod.model.trainable_weights])
        self.assertEqual(159500, param_count)

    def test_intermediate_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='intermediate_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        param_count = np.sum([count_params(w) for w in mod.model.trainable_weights])
        self.assertEqual(159500, param_count)

    def test_late_integration_model_builds(self):
        # Arrange
        mod = self._sut(integration_type='late_integration')

        # Act
        mod.build_model()

        # Assert
        self.assertIsInstance(mod, MultisensoryBase)
        param_count = np.sum([count_params(w) for w in mod.model.trainable_weights])
        self.assertEqual(159500, param_count)
