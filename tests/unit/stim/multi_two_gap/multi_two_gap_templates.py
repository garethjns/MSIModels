import unittest

from msi_models.stim.multi_two_gap.multi_two_gap_params import MultiTwoGapParams
from msi_models.stim.multi_two_gap.multi_two_gap_templates import template_matched, template_unmatched, template_sync


class TestTemplateUnmatched(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = template_unmatched

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut(duration_tol=0.1)
        self.assertIsInstance(params, MultiTwoGapParams)


class TestTemplateMatched(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = template_matched

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut(duration_tol=0.1)
        self.assertIsInstance(params, MultiTwoGapParams)


class TestTemplateSync(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = template_sync

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut()
        self.assertIsInstance(params, MultiTwoGapParams)

    def test_modified_params_raise_no_validation_errors(self):
        params = self._sut(duration_tol=0.1)
        self.assertIsInstance(params, MultiTwoGapParams)
