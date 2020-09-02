import os

import h5py
import numpy as np
import pandas as pd


class DataFixture:
    path: str = 'test_data_fixture.hdf5'
    x_left = np.expand_dims(np.array([[0.1, 0, 0.1],
                                      [0, 0.1, 0],
                                      [0.1, 0.1, 0]]),
                            axis=2)
    x_mask_left = np.expand_dims(np.array([[1, 0, 1],
                                           [0, 1, 0],
                                           [1, 1, 0]]),
                                 axis=2)

    x_right = np.expand_dims(np.array([[0.1, 0, 0.1],
                                       [0, 0.1, 0],
                                       [0.1, 0.1, 0]]),
                             axis=2)
    x_mask_right = np.expand_dims(np.array([[1, 0, 1],
                                            [0, 1, 0],
                                            [1, 1, 0]]),
                                  axis=2)

    y_rate_left = np.array([2, 1, 2])
    y_rate_right = np.array([2, 1, 2])
    y_dec_left = np.array([1, 0, 1])
    y_dec_right = np.array([1, 0, 1])

    def clear(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
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
            f.create_dataset("x", data=self.x_left, compression='gzip')
            f.create_dataset("x_mask", data=self.x_mask_left, compression='gzip')
            f.create_dataset("y_rate", data=self.y_rate_left, compression='gzip')
            f.create_dataset("y_dec", data=self.y_dec_left, compression='gzip')


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
    /--agg/
           |--y_rate
           |--y_dec
   |--summary

    """
    path: str = 'test_multisensory_data_fixture.hdf5'

    def save(self):
        with h5py.File(self.path, 'w') as f:
            f.create_dataset("left/x", data=self.x_left, compression='gzip')
            f.create_dataset("left/x_mask", data=self.x_mask_left, compression='gzip')
            f.create_dataset("left/y_rate", data=self.y_rate_left, compression='gzip')
            f.create_dataset("left/y_dec", data=self.y_dec_left, compression='gzip')
            f.create_dataset("right/x", data=self.x_right, compression='gzip')
            f.create_dataset("right/x_mask", data=self.x_mask_right, compression='gzip')
            f.create_dataset("right/y_rate", data=self.y_rate_right, compression='gzip')
            f.create_dataset("right/y_dec", data=self.y_dec_right, compression='gzip')
            f.create_dataset("agg/y_rate", data=(self.y_rate_left + self.y_rate_right) / 2, compression='gzip')
            f.create_dataset("agg/y_dec", data=(self.y_dec_left + self.y_dec_right) / 2, compression='gzip')

        summary = pd.DataFrame({'example': [0, 1, 2]})
        summary.to_hdf(self.path, key='summary', mode='a')
