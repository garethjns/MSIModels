import unittest

from msi_models.exceptions.params import InvalidParameterException
from msi_models.stimset.channel_config import ChannelConfig
from tests.common.fixtures.data_fixtures import UnisensoryDataFixture


class TestChannelConfig(unittest.TestCase):
    _data_fixture = UnisensoryDataFixture()
    _sut = ChannelConfig

    @classmethod
    def setUpClass(cls):
        cls._data_fixture.save()

    @classmethod
    def tearDownClass(cls):
        cls._data_fixture.clear()

    def test_valid_config_survives_pydantic_validation(self):
        # Act
        chan_config = self._sut(path=self._data_fixture.path,
                                x_keys=['x', 'x_mask'],
                                y_keys=['y_rate', 'y_dec'])

        # Assert
        self.assertIsInstance(chan_config, ChannelConfig)

    def test_invalid_x_keys_fails_pydantic_validation(self):
        # Actert
        self.assertRaises(InvalidParameterException,
                          lambda: self._sut(path=self._data_fixture.path,
                                            x_keys=['x_nope', 'x_mask'],
                                            y_keys=['y_rate', 'y_dec']))

    def test_invalid_y_keys_fails_pydantic_validation(self):
        # Actert
        self.assertRaises(InvalidParameterException,
                          lambda: self._sut(path=self._data_fixture.path,
                                            x_keys=['x', 'x_mask'],
                                            y_keys=['y_nope', 'y_dec']))
