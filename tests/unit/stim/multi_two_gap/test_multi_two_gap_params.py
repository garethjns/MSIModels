import unittest
from functools import partial
from unittest.mock import MagicMock

from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapParams


class TestMultiTwoGapParams(unittest.TestCase):
    _sut = MultiTwoGapParams

    def setUp(self) -> None:
        self.common_kwargs = {'gap_1': MagicMock(spec=partial),
                              'gap_2': MagicMock(spec=partial),
                              'background': MagicMock(spec=partial),
                              'event': MagicMock(spec=partial),
                              'cache': True,
                              'duration_tol': 1,
                              'duration': 1000,
                              'background_weight': 0.01}

        self.less_common_kwargs = {"n_events": 10,
                                   "seed": 1}

    def test_validation_sync_fails_for_async_stim_because_seeds(self):
        # Actsert
        self.assertRaises(IncompatibleParametersException,
                          lambda: self._sut(n_channels=2,
                                            n_events=10,
                                            seed=[1, 2],
                                            validate_as_sync=True,
                                            validate_as_matched=True,
                                            **self.common_kwargs))

    def test_validation_sync_fails_for_async_stim_because_events(self):
        # Actsert
        self.assertRaises(IncompatibleParametersException,
                          lambda: self._sut(n_channels=2,
                                            n_events=[10, 11],
                                            seed=1,
                                            validate_as_sync=True,
                                            validate_as_matched=True,
                                            **self.common_kwargs))

    def test_validation_sync_correctly_ignored_when_off(self):
        # Act
        multi_params = self._sut(n_channels=2,
                                 n_events=10,
                                 seed=[1, 2],
                                 validate_as_sync=None,
                                 validate_as_matched=True,
                                 **self.common_kwargs)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)

    def test_validation_sync_as_false_for_async_stim(self):
        # Act
        multi_params = self._sut(n_channels=2,
                                 n_events=11,
                                 seed=[1, 2],
                                 validate_as_sync=False,
                                 validate_as_matched=True,
                                 **self.common_kwargs)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)

    def test_validation_matched_fails_for_unmatched_stim(self):
        # Actsert
        self.assertRaises(IncompatibleParametersException,
                          lambda: self._sut(n_channels=2,
                                            n_events=[10, 11],
                                            seed=1,
                                            validate_as_sync=False,
                                            validate_as_matched=True,
                                            **self.common_kwargs))

    def test_validation_matched_correctly_ignored_when_off(self):
        # Act
        multi_params = self._sut(n_channels=2,
                                 n_events=[10, 11],
                                 seed=1,
                                 validate_as_sync=False,
                                 validate_as_matched=None,
                                 **self.common_kwargs)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)

    def test_validation_matched_as_false_for_unmatched_stim(self):
        # Act
        multi_params = self._sut(n_channels=2,
                                 n_events=[10, 11],
                                 seed=1,
                                 validate_as_sync=False,
                                 validate_as_matched=False,
                                 **self.common_kwargs)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)

    def test_init_works_without_lists(self):
        # Arrange
        params = self.common_kwargs
        params.update(self.less_common_kwargs)

        # Act
        multi_params = self._sut(n_channels=2,
                                 validate_as_sync=None,
                                 validate_as_matched=None,
                                 **params)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)
        self.assertEqual(len(multi_params.n_events), 2)

    def test_init_works_with_lists(self):
        # Arrange
        params = {k: [v, v] for k, v in self.common_kwargs.items()}
        params.update({k: [v, v] for k, v in self.less_common_kwargs.items()})

        # Act
        multi_params = self._sut(n_channels=2,
                                 validate_as_sync=None,
                                 validate_as_matched=None,
                                 **params)

        # Assert
        self.assertIsInstance(multi_params, MultiTwoGapParams)
        self.assertEqual(len(multi_params.n_events), 2)

    def test_unexpected_number_of_channels_raises_error(self):
        # Arrange
        params = {k: [v, v, v] for k, v in self.common_kwargs.items()}
        params.update({k: [v, v, v] for k, v in self.less_common_kwargs.items()})

        # Act
        self.assertRaises(IncompatibleParametersException,
                          lambda: self._sut(n_channels=2,
                                            validate_as_sync=None,
                                            validate_as_matched=None,
                                            **params))
