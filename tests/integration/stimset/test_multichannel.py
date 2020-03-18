import unittest
from unittest.mock import MagicMock
import copy
from msi_models.exceptions.params import InvalidParameterException, IncompatibleParametersException
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel, MultiChannelConfig
from tests.common.fixtures.data_fixtures import MultisensoryDataFixture
import numpy as np


class TestMultiChannelConfig(unittest.TestCase):
    _data_fixture = MultisensoryDataFixture()
    _sut = MultiChannelConfig

    @classmethod
    def setUpClass(cls):
        cls._data_fixture.save()

    @classmethod
    def tearDownClass(cls):
        cls._data_fixture.clear()

    def setUp(self):
        # Specifying ChannelConfig is necessary here, as Pydantic expects the ChannelConfig type, and will additionally
        # run the ChannelConfig checks on the provided object.
        mock_config = MagicMock(spec=ChannelConfig)
        mock_config.path = self._data_fixture.path
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
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right])

        # Assert
        self.assertIsInstance(multi_config, MultiChannelConfig)

    def test_invalid_train_props_fails_pydantic_validation(self):
        # Arrange
        # Use real config for this
        common_kwargs = {"path": self._data_fixture.path,
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
                          lambda: self._sut(channels=[left_config, right_config]))

    def test_unspecified_seed_is_set_to_int(self):
        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right])

        # Assert
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)

    def test_none_seed_is_set_to_int(self):
        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 seed=None)

        # Assert
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)

    def test_int_seed_is_not_changed(self):
        # Arrange
        seed = 123

        # Act
        multi_config = self._sut(channels=[self.chan_config_left, self.chan_config_right],
                                 seed=seed)

        # Assert
        self.assertEqual(seed, multi_config.seed)
        self.assertEqual(multi_config.seed, multi_config.channels[0].seed)
        self.assertEqual(multi_config.seed, multi_config.channels[1].seed)


class TestMultiChannel(unittest.TestCase):
    _data_fixture = MultisensoryDataFixture()
    _sut = MultiChannel

    @classmethod
    def setUpClass(cls):
        cls._data_fixture.save()

    def setUp(self):
        common_kwargs = {"path": self._data_fixture.path,
                         "train_prop": 0.8,
                         "x_keys": ["x", "x_mask"],
                         "y_keys": ["y_rate", "y_dec"],
                         "seed": 100}

        left_config = ChannelConfig(key='left', **common_kwargs)
        right_config = ChannelConfig(key='right', **common_kwargs)

        self.multi_config = MultiChannelConfig(channels=[left_config, right_config])

    @classmethod
    def tearDownClass(cls):
        cls._data_fixture.clear()

    def test_init_from_config(self):
        # Act
        multi_channel = self._sut(self.multi_config)

        # Assert
        self.assertIsInstance(multi_channel, MultiChannel)

    def test_seeds_match_between_channels(self):
        # Act
        multi_channel = self._sut(self.multi_config)

        # Assert
        self.assertEqual(multi_channel.channels[0].config.seed, multi_channel.channels[1].config.seed)

    def test_train_and_test_indexes_match_between_channels(self):
        # Act
        multi_channel = self._sut(self.multi_config)
        _ = multi_channel.x_train

        # Assert
        self.assertListEqual(list(multi_channel.channels[0].train_idx),
                             list(multi_channel.channels[1].train_idx))

    def test_channel_gets_xs_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        x = multi_channel.x

        # Assert
        self.assertIsInstance(x, dict)
        self.assertListEqual(['left_x', 'left_x_mask', 'right_x', 'right_x_mask'], list(x.keys()))
        self.assertEqual((3, 3, 1), x['left_x'].shape)
        self.assertEqual((3, 3, 1), x['left_x_mask'].shape)
        self.assertEqual((3, 3, 1), x['right_x'].shape)
        self.assertEqual((3, 3, 1), x['right_x_mask'].shape)

    def test_channel_gets_ys_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        y = multi_channel.y

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(['y_dec', 'y_rate'], list(y.keys()))
        self.assertEqual((3,), y['y_rate'].shape)
        self.assertEqual((3,), y['y_dec'].shape)

    def test_channel_gets_xs_train_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        x = multi_channel.x_train

        # Assert
        self.assertIsInstance(x, dict)
        self.assertListEqual(['left_x', 'left_x_mask', 'right_x', 'right_x_mask'], list(x.keys()))
        self.assertEqual((2, 3, 1), x['left_x'].shape)
        self.assertEqual((2, 3, 1), x['left_x_mask'].shape)
        self.assertEqual((2, 3, 1), x['right_x'].shape)
        self.assertEqual((2, 3, 1), x['right_x_mask'].shape)
        # Test fixture contains same data on each channel
        self.assertTrue(np.all(x['left_x'] == x['left_x']))
        self.assertTrue(np.all(x['left_x_mask'] == x['left_x_mask']))

    def test_channel_gets_ys_train_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        y = multi_channel.y_train

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(['y_dec', 'y_rate'], list(y.keys()))
        self.assertEqual((2,), y['y_rate'].shape)
        self.assertEqual((2,), y['y_dec'].shape)

    def test_channel_gets_xs_test_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        x = multi_channel.x_test

        # Assert
        self.assertIsInstance(x, dict)
        self.assertListEqual(['left_x', 'left_x_mask', 'right_x', 'right_x_mask'], list(x.keys()))
        self.assertEqual((1, 3, 1), x['left_x'].shape)
        self.assertEqual((1, 3, 1), x['left_x_mask'].shape)
        self.assertEqual((1, 3, 1), x['right_x'].shape)
        self.assertEqual((1, 3, 1), x['right_x_mask'].shape)
        # Test fixture contains same data on each channel
        self.assertTrue(np.all(x['left_x'] == x['left_x']))
        self.assertTrue(np.all(x['left_x_mask'] == x['left_x_mask']))

    def test_channel_gets_ys_test_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        y = multi_channel.y_test

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(['y_dec', 'y_rate'], list(y.keys()))
        self.assertEqual((1,), y['y_rate'].shape)
        self.assertEqual((1,), y['y_dec'].shape)