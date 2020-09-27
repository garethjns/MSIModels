"""Example training multisensory model."""
import os

from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stimset.channel_config import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannel
from msi_models.stimset.multi_channel_config import MultiChannelConfig
from msi_models.tf_helpers import limit_gpu_memory

if __name__ == "__main__":
    limit_gpu_memory(5000)

    # Specify a dataset previously generated by scripts/mixed_type_multisensory_data.py
    path = 'data/scripts_unisensory_data_hard.hdf5'
    if not os.path.exists(path):
        raise RuntimeWarning(f"Need to generate data first - run scripts/mixed_type_multisensory_data.py")

    common_kwargs = {"path": path, "train_prop": 0.8, "seed": 100,
                     "x_keys": ["x", "x_mask"], "y_keys": ["y_rate", "y_dec"]}

    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)
    multi_config = MultiChannelConfig(path=path, key='agg', y_keys=common_kwargs["y_keys"],
                                      channels=[left_config, right_config])
    mc = MultiChannel(multi_config)

    mc.plot_example()
    mc.plot_example()

    mod = MultisensoryClassifier(integration_type='intermediate_integration', opt='adam', batch_size=10000, lr=0.004)

    y_names = ['agg_y_rate', 'agg_y_dec']
    mod.fit(mc.x_train, {k: v for k, v in mc.y_train.items() if k in y_names}, epochs=1200, validation_split=0.4)

    mod.predict(mc.x_train[0:10])
