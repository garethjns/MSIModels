import os

from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

if __name__ == "__main__":
    # Set path to .hdf file
    fn = 'data/sample_multisensory_data_sync_250k.hdf5'
    fn = 'data/sample_multisensory_data_mix_hard_10.hdf5'
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    # Create channel configs to load for /left/ and /right/ keys
    common_kwargs = {"path": path,
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}
    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)

    # Combine the channels and define aggregate (/agg/) key to use as y(s)
    multi_config = MultiChannelConfig(path=path,
                                      key='agg',
                                      y_keys=["y_rate", "y_dec"],
                                      channels=[left_config, right_config])
    mc = MultiChannel(multi_config)

    # View some examples
    mc.plot_example()
    mc.plot_example()