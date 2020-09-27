from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory

N_DATA_ROWS = 100000
N_REPS = 10
N_EPOCHS = 2000

DATASETS = [ExperimentalDataset("easy", n=N_DATA_ROWS, difficulty=15).build("data/contained_mix_easy.hdf5"),
            ExperimentalDataset("medium", n=N_DATA_ROWS, difficulty=25).build("data/contained_mix_medium.hdf5"),
            ExperimentalDataset("hard", n=N_DATA_ROWS, difficulty=35).build("data/contained_mix_hard.hdf5")]
MODELS = ['early_integration', 'intermediate_integration', 'late_integration']


def add_config_data(exp: Experiment):
    for exp_data in DATASETS:
        exp.add_data(exp_data)


def add_config_models(exp: Experiment):
    common_model_kwargs = {'opt': 'adam', 'batch_size': int(min(N_DATA_ROWS / 10, 15000)), 'lr': 0.01}
    for int_type in MODELS:
        mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs),
                                name=int_type)
        exp.add_model(mod)


if __name__ == "__main__":
    limit_gpu_memory(5000)

    exp = Experiment(name='contained_experiment', n_epochs=N_EPOCHS, n_reps=N_REPS)
    add_config_data(exp)
    add_config_models(exp)
    exp.run()
