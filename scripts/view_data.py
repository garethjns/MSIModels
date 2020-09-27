import os

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel
from msi_models.stimset.multi_channel_config import MultiChannelConfig

if __name__ == "__main__":
    # Set path to .hdf file
    path = "data/example_mix_easy.hdf5"
    if not os.path.exists(path):
        ExperimentalDataset("scripts_example_easy", n=1000, difficulty=12).build(path)

    # Create channel configs to load for /left/ and /right/ keys
    common_kwargs = {"path": path, "train_prop": 0.8, "seed": 100,
                     "x_keys": ["x", "x_mask"], "y_keys": ["y_rate", "y_dec"]}
    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)

    # Combine the channels and define aggregate (/agg/) key to use as y(s)
    multi_config = MultiChannelConfig(path=path, key='agg', y_keys=["y_rate", "y_dec"],
                                      channels=[left_config, right_config])
    mc = MultiChannel(multi_config)

    # View some examples
    mc.plot_example()
    mc.plot_example()

    # View summary
    mc.plot_summary()
