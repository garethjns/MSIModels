"""Example training a model using the experimental wrappers"""
import os

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig

if __name__ == "__main__":
    # Prepare data
    fn = "data/multisensory_data_matched.hdf5"
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    common_kwargs = {"path": path,
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"],
                     "seed": 100}

    left_config = ChannelConfig(key='left', **common_kwargs)
    right_config = ChannelConfig(key='right', **common_kwargs)
    multi_config = MultiChannelConfig(path=path,
                                      key='agg',
                                      y_keys=["y_rate", "y_dec"],
                                      channels=[left_config, right_config])

    exp_data = ExperimentalDataset(name='example_dataset',
                                   config=multi_config)
    exp_data.build(seed=123)

    # Prepare model
    mod = MultisensoryClassifier(integration_type='intermediate_integration',
                                 opt='adam',
                                 epochs=1000,
                                 batch_size=2000,
                                 lr=0.0025)
    exp_model = ExperimentalModel(model=mod)

    # Fit
    exp_model.fit(exp_data,
                  batch_size=500,
                  epochs=1000)

    # Evaluate
    exp_model.plot_example(exp_data,
                           dec_key='agg_y_dec')
    train_report, test_report = exp_model.report(exp_data)

    exp_model.plot_example(exp_data,
                           dec_key='agg_y_dec')
    exp_model.plot_example(exp_data,
                           dec_key='agg_y_dec',
                           mistake=True)
