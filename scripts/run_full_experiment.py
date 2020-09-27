from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory

N_DATA_ROWS = 1000
N_REPS = 5
N_EPOCHS = 2000

if __name__ == "__main__":
    limit_gpu_memory(5000)

    # Prepare experiment
    exp = Experiment(name='scripts_example_experiment', n_epochs=N_EPOCHS, n_reps=N_REPS)

    # Prepare and add data (data doesn't need to have been pre-generated)
    for exp_data in [ExperimentalDataset("scripts_example_easy",
                                         n=N_DATA_ROWS, difficulty=12).build("data/scripts_example_mix_easy.hdf5"),
                     ExperimentalDataset("scripts_example_hard",
                                         n=N_DATA_ROWS, difficulty=12).build("data/scripts_example_mix_hard.hdf5")]:
        exp.add_data(exp_data)

    # Prepare and add models
    common_model_kwargs = {'opt': 'adam', 'batch_size': int(min(N_DATA_ROWS / 10, 15000)), 'lr': 0.01}
    for int_type in ['early_integration', 'intermediate_integration', 'late_integration']:
        mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs), name=int_type)
        exp.add_model(mod)

    # Run experiment
    exp.run()
