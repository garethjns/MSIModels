import os

from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig

if __name__ == "__main__":
    path = os.getcwd().split('msi_models')[0].replace('\\', '/')

    # Prepare runs
    common_data_kwargs = {"train_prop": 0.8,
                          "x_keys": ["x", "x_mask"],
                          "y_keys": ["y_rate", "y_dec"],
                          "seed": 100}

    fns = {"multi_sync": "data/sample_multisensory_data_sync.hdf5",
           "multi_matched": "data/sample_multisensory_data_matched.hdf5",
           "multi_unmatched": "data/sample_multisensory_data_unmatched.hdf5"}

    exp_datasets = []
    for fk, fn in fns.items():
        exp_datasets.append(
            ExperimentalDataset(name=fk,
                                config=MultiChannelConfig(path=f"{path}/{fn}",
                                                          key='agg',
                                                          y_keys=["y_rate", "y_dec"],
                                                          channels=[ChannelConfig(path=f"{path}/{fn}",
                                                                                  key='left', **common_data_kwargs),
                                                                    ChannelConfig(path=f"{path}/{fn}",
                                                                                  key='right', **common_data_kwargs)])))

    exp_models = []
    for mod in ['early_integration', 'intermediate_integration', 'late_integration']:
        exp_models.append(
            ExperimentalModel(model=MultisensoryClassifier(integration_type='intermediate_integration',
                                                           opt='adam',
                                                           epochs=10,
                                                           batch_size=2000,
                                                           lr=0.0025)))

    experiment = Experiment(models=exp_models,
                            datasets=exp_datasets,
                            name="example_experiment2")

    experiment.run()
