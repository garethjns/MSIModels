import unittest

import numpy as np

from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel
from msi_models.stimset.multi_channel_config import MultiChannelConfig
from tests.common.fixtures.data_fixtures import MultisensoryDataFixture


class TestMultiChannel(unittest.TestCase):
    _data_fixture = MultisensoryDataFixture()
    _sut = MultiChannel

    @classmethod
    def setUpClass(cls):
        cls._data_fixture.save()

    def setUp(self):
        # For single chans
        common_kwargs = {"path": self._data_fixture.path,
                         "train_prop": 0.8,
                         "x_keys": ["x", "x_mask"],
                         "y_keys": ["y_rate", "y_dec"],
                         "seed": 100}

        left_config = ChannelConfig(key='left', **common_kwargs)
        right_config = ChannelConfig(key='right', **common_kwargs)

        self.multi_config = MultiChannelConfig(channels=[left_config, right_config],
                                               path=self._data_fixture.path,
                                               y_keys=['y_rate', 'y_dec'])

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
        self.assertEqual(multi_channel.channels[0].config.seed, multi_channel.y_channel.config.seed)

    def test_train_and_test_indexes_match_between_channels(self):
        # Act
        multi_channel = self._sut(self.multi_config)

        # Assert
        self.assertListEqual(list(multi_channel.channels[0].train_idx),
                             list(multi_channel.channels[1].train_idx))
        self.assertListEqual(list(multi_channel.channels[0].train_idx),
                             list(multi_channel.y_channel.train_idx))

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
        self.assertListEqual(['agg_y_rate', 'agg_y_dec', 'left_y_rate', 'left_y_dec', 'right_y_rate', 'right_y_dec'],
                             list(y.keys()))
        self.assertEqual((3,), y['agg_y_rate'].shape)
        self.assertEqual((3,), y['agg_y_dec'].shape)
        self.assertEqual((3,), y['left_y_rate'].shape)
        self.assertEqual((3,), y['left_y_dec'].shape)
        self.assertEqual((3,), y['right_y_rate'].shape)
        self.assertEqual((3,), y['right_y_dec'].shape)

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
        self.assertTrue(np.all(x['left_x'] == x['right_x']))
        self.assertTrue(np.all(x['left_x_mask'] == x['right_x_mask']))

    def test_channel_gets_ys_train_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        y = multi_channel.y_train

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(['agg_y_rate', 'agg_y_dec', 'left_y_rate', 'left_y_dec', 'right_y_rate', 'right_y_dec'],
                             list(y.keys()))
        self.assertEqual((2,), y['agg_y_rate'].shape)
        self.assertEqual((2,), y['agg_y_dec'].shape)
        self.assertEqual((2,), y['left_y_rate'].shape)
        self.assertEqual((2,), y['left_y_dec'].shape)
        self.assertEqual((2,), y['right_y_rate'].shape)
        self.assertEqual((2,), y['right_y_dec'].shape)

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
        self.assertTrue(np.all(x['left_x'] == x['right_x']))
        self.assertTrue(np.all(x['left_x_mask'] == x['right_x_mask']))

    def test_channel_gets_ys_test_correctly(self):
        # Arrange
        multi_channel = self._sut(self.multi_config)

        # Act
        y = multi_channel.y_test

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(['agg_y_rate', 'agg_y_dec', 'left_y_rate', 'left_y_dec', 'right_y_rate', 'right_y_dec'],
                             list(y.keys()))
        self.assertEqual((1,), y['agg_y_rate'].shape)
        self.assertEqual((1,), y['agg_y_dec'].shape)
        self.assertEqual((1,), y['left_y_rate'].shape)
        self.assertEqual((1,), y['left_y_dec'].shape)
        self.assertEqual((1,), y['right_y_rate'].shape)
        self.assertEqual((1,), y['right_y_dec'].shape)
