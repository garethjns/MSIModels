import unittest
from unittest.mock import MagicMock

from msi_models.stimset.channel import Channel
from tests.common.fixtures.data_fixtures import UnisensoryDataFixture


class TestChannel(unittest.TestCase):
    _data_fixture = UnisensoryDataFixture()
    _sut = Channel

    @classmethod
    def setUpClass(cls):
        cls._data_fixture.save()

    def setUp(self):
        self._mock_config = MagicMock()
        self._mock_config.path = self._data_fixture.path
        self._mock_config.seed = None
        self._mock_config.train_prop = 0.8
        self._mock_config.key = ''
        self._mock_config.x_keys = ['x', 'x_mask']
        self._mock_config.y_keys = ['y_rate', 'y_dec']

    @classmethod
    def tearDownClass(cls):
        cls._data_fixture.clear()

    def test_init_from_config_with_no_key_specified(self):
        # Act
        channel = self._sut(self._mock_config)

        # Assert
        self.assertIsInstance(channel, Channel)

    def test_init_from_config_with_root_key_specified(self):
        # Arrange
        self._mock_config.key = '/'

        # Act
        channel = self._sut(self._mock_config)

        # Assert
        self.assertIsInstance(channel, Channel)

    def test_channel_gets_x_correctly(self):
        # Arrange
        channel = self._sut(self._mock_config)

        # Act
        x = channel.x

        # Assert
        self.assertIsInstance(x, dict)
        self.assertListEqual(list(x.keys()), ['x', 'x_mask'])
        self.assertEqual((3, 3, 1), x['x'].shape)
        self.assertEqual((3, 3, 1), x['x_mask'].shape)

    def test_channel_gets_y_correctly(self):
        # Arrange
        channel = self._sut(self._mock_config)

        # Act
        y = channel.y

        # Assert
        self.assertIsInstance(y, dict)
        self.assertListEqual(list(y.keys()), ['y_rate', 'y_dec'])
        self.assertEqual((3,), y['y_rate'].shape)
        self.assertEqual((3,), y['y_dec'].shape)

    def test_channel_gets_x_train_correctly(self):
        # Arrange
        channel = self._sut(self._mock_config)

        # Act
        x_train = channel.x_train

        # Assert
        self.assertIsInstance(x_train, dict)
        self.assertListEqual(list(x_train.keys()), ['x', 'x_mask'])
        self.assertEqual((2, 3, 1), x_train['x'].shape)
        self.assertEqual((2, 3, 1), x_train['x_mask'].shape)

    def test_channel_gets_y_train_correctly(self):
        # Arrange
        channel = self._sut(self._mock_config)

        # Act
        y_train = channel.y_train

        # Assert
        self.assertIsInstance(y_train, dict)
        self.assertListEqual(list(y_train.keys()), ['y_rate', 'y_dec'])
        self.assertEqual((2,), y_train['y_rate'].shape)
        self.assertEqual((2,), y_train['y_dec'].shape)

    def test_channel_gets_x_test_correctly(self):
        # Arrange
        channel = self._sut(self._mock_config)

        # Act
        x_test = channel.x_test

        # Assert
        self.assertIsInstance(x_test, dict)
        self.assertListEqual(list(x_test.keys()), ['x', 'x_mask'])
        self.assertEqual((1, 3, 1), x_test['x'].shape)
        self.assertEqual((1, 3, 1), x_test['x_mask'].shape)

    def test_channel_gets_y_train_correctly_smaller_train_prop(self):
        # Arrange
        self._mock_config.train_prop = 0.2
        channel = self._sut(self._mock_config)

        # Act
        y_test = channel.y_train

        # Assert
        self.assertIsInstance(y_test, dict)
        self.assertListEqual(list(y_test.keys()), ['y_rate', 'y_dec'])
        self.assertEqual((1,), y_test['y_rate'].shape)
        self.assertEqual((1,), y_test['y_dec'].shape)

    def test_channel_gets_y_test_correctly_smaller_train_prop(self):
        # Arrange
        self._mock_config.train_prop = 0.2
        channel = self._sut(self._mock_config)

        # Act
        y_test = channel.y_test

        # Assert
        self.assertIsInstance(y_test, dict)
        self.assertListEqual(list(y_test.keys()), ['y_rate', 'y_dec'])
        self.assertEqual((2,), y_test['y_rate'].shape)
        self.assertEqual((2,), y_test['y_dec'].shape)
