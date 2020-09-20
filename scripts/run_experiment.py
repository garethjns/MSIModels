import os

import tensorflow as tf

from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=5000)])

N_REPS = 5
N_EPOCHS = 2000

if __name__ == "__main__":
    # Prepare data
    fn = "data/sample_multisensory_data_mix_hard_250k.hdf5"
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    # Prepare experiment
    exp = Experiment(name='example_experiment', n_epochs=N_EPOCHS, n_reps=N_REPS)

    # Add data
    exp.add_data(path)

    # Prepare and add models
    common_model_kwargs = {'opt': 'adam', 'batch_size': 15000, 'lr': 0.01}
    for int_type in ['early_integration', 'intermediate_integration', 'late_integration']:
        mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs), name=int_type)
        exp.add_model(mod)

    # Run experiment
    exp.run()
