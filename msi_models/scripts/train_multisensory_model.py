"""Example training unisensory model."""

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.stimset.channel import ChannelConfig
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel


if __name__ == "__main__":
    common_kwargs = {"path": "data/multisensory_data.hdf5",
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}

    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)
    multi_config = MultiChannelConfig(channels=[left_config, right_config])

    mc = MultiChannel(multi_config)

    mc.plot_example()
    mc.plot_example()

    exp_model = ExperimentalModel(data=mc,
                                  model=MultisensoryClassifier(integration_type='intermediate_integration',
                                                               opt='adam',
                                                               epochs=1000,
                                                               batch_size=2000,
                                                               lr=0.0025))
    exp_model.fit()
    exp_model.evaluate()
    exp_model.plot_example()
    train_report, test_report = exp_model.report()

    exp_model.plot_example()
    exp_model.plot_example(mistake=True)
