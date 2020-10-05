import copy
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel_config import MultiChannelConfig
from tests.common.fixtures.data_fixtures import MultisensoryDataFixture


class TestMultiChannelConfig(unittest.TestCase):
    _data_fixture = MultisensoryDataFixture()
    _sut = MultiChannelConfig

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._data_fixture.save(path=cls._tmp_dir.name)
        cls._data_fixture_path = os.path.join(cls._tmp_dir.name, cls._data_fixture.path)

    @classmethod
    def tearDownClass(cls):
        cls._data_fixture.clear()
        cls._tmp_dir.cleanup()

    def setUp(self):
        # Specifying ChannelConfig is necessary here, as Pydantic expects the ChannelConfig type, and will additionally
        # run the ChannelConfig checks on the provided object.
        mock_config = MagicMock(spec=ChannelConfig)
        mock_config.path = self._data_fixture_path
        mock_config.seed = None
        mock_config.train_prop = 0.8
        mock_config.key = ''
        mock_config.x_keys = ['x', 'x_mask']
        mock_config.y_keys = ['y_rate', 'y_dec']

        self.chan_config_left = copy.deepcopy(mock_config)
        self.chan_config_left.key = 'left'
        self.chan_config_right = copy.deepcopy(mock_config)
        self.chan_config_right.key = 'right'

    def test_valid_config_survives_pydantic_validation(self):
        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 path=self._data_fixture_path,
                                 y_keys=['y_rate', 'y_dec'])

        # Assert
        self.assertIsInstance(multi_config, MultiChannelConfig)

    def test_invalid_train_props_fails_pydantic_validation(self):
        # Arrange
        # Use real config for this
        common_kwargs = {"path": self._data_fixture_path,
                         "x_keys": ["x", "x_mask"],
                         "y_keys": ["y_rate", "y_dec"],
                         "seed": 100}

        left_config = ChannelConfig(key='left',
                                    train_prop=0.3,
                                    **common_kwargs)
        right_config = ChannelConfig(key='right',
                                     train_prop=0.7,
                                     **common_kwargs)

        # Actsert
        self.assertRaises(IncompatibleParametersException,
                          lambda: self._sut(channels=[left_config, right_config],
                                            path=self._data_fixture.path,
                                            y_keys=['y_rate', 'y_dec']))

    def test_unspecified_seed_is_set_to_int(self):
        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 path=self._data_fixture_path,
                                 y_keys=['y_rate', 'y_dec'])

        # Assert
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)

    def test_none_seed_is_set_to_int(self):
        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 seed=None,
                                 path=self._data_fixture_path,
                                 y_keys=['y_rate', 'y_dec'])

        # Assert
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)

    def test_int_seed_is_not_changed(self):
        # Arrange
        seed = 123

        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 seed=seed,
                                 path=self._data_fixture_path,
                                 y_keys=['y_rate', 'y_dec'])

        # Assert
        self.assertEqual(seed, multi_config.seed)
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)
