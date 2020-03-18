import h5py
import os

import numpy as np


class DataFixture:
    path: str = 'test_data_fixture.hdf5'
    x = np.expand_dims(np.array([[0.1, 0, 0.1],
                                 [0, 0.1, 0],
                                 [0.1, 0.1, 0]]),
                       axis=2)
    x_mask = np.expand_dims(np.array([[1, 0, 1],
                                      [0, 1, 0],
                                      [1, 1, 0]]),
                            axis=2)
    y_rate = np.array([2, 1, 2])
    y_dec = np.array([1, 0, 1])

    def clear(self):
        try:
            os.remove(self.path)
        except Exception:
            pass


class UnisensoryDataFixture(DataFixture):
    """
    Mocks hdf5 file with following structure:

    |--x (n, duration_pts, 1)
    |--x_indicators (n, duration_pts, 1)
    |--y_rate (n,)
    |--y_dec (n,)

    """
    path: str = 'test_unisensory_data_fixture.hdf5'

    def save(self):
        with h5py.File(self.path, 'w') as f:
            f.create_dataset("x", data=self.x, compression='gzip')
            f.create_dataset("x_mask", data=self.x_mask, compression='gzip')
            f.create_dataset("y_rate", data=self.y_rate, compression='gzip')
            f.create_dataset("y_dec", data=self.y_dec, compression='gzip')


class MultisensoryDataFixture(DataFixture):
    """
    Mocks hdf5 file with following structure:

    |--left/
            |--x (n, duration_pts, 1)
            |--x_indicators (n, duration_pts, 1)
            |--y_rate (n,)
            |--y_dec (n,)
            |--configs (not added)
    |--right/
            |--x (n, duration_pts, 1)
            |--x_indicators (n, duration_pts, 1)
            |--y_rate (n,)
            |--y_dec (n,)
            |--configs (not added)

    """
    path: str = 'test_multisensory_data_fixture.hdf5'

    def save(self):
        with h5py.File(self.path, 'w') as f:
            f.create_dataset("left/x", data=self.x, compression='gzip')
            f.create_dataset("left/x_mask", data=self.x_mask, compression='gzip')
            f.create_dataset("left/y_rate", data=self.y_rate, compression='gzip')
            f.create_dataset("left/y_dec", data=self.y_dec, compression='gzip')
            f.create_dataset("right/x", data=self.x, compression='gzip')
            f.create_dataset("right/x_mask", data=self.x_mask, compression='gzip')
            f.create_dataset("right/y_rate", data=self.y_rate, compression='gzip')
            f.create_dataset("right/y_dec", data=self.y_dec, compression='gzip')
