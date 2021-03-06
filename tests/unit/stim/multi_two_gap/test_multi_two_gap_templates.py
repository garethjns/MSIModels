import unittest

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate


class TestTemplateUnmatched(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = MultiTwoGapTemplate['unmatched_async']

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut.set_options().build()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut.set_options(duration_tol=0.1).build()
        self.assertIsInstance(params, MultiTwoGapParams)


class TestTemplateMatched(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = MultiTwoGapTemplate['matched_async']

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut.set_options().build()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut.set_options(duration_tol=0.1).build()
        self.assertIsInstance(params, MultiTwoGapParams)


class TestTemplateSync(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = MultiTwoGapTemplate['matched_sync']

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut.set_options().build()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut.set_options(duration_tol=0.1).build()
        self.assertIsInstance(params, MultiTwoGapParams)

