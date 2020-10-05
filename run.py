from msi_models.experiment.experiment import Experiment
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory

N_DATA_ROWS = 50000
N_REPS = 3
N_JOBS = 28
N_EPOCHS = 2000
EXP_PREFIX = 'contained'

DATASETS = [ExperimentalDataset("easy", n=N_DATA_ROWS,
                                difficulty=17).build(f"data/{EXP_PREFIX}_mix_easy_{N_DATA_ROWS}.hdf5", n_jobs=N_JOBS),
            ExperimentalDataset("medium", n=N_DATA_ROWS,
                                difficulty=27).build(f"data/{EXP_PREFIX}_mix_medium_{N_DATA_ROWS}.hdf5", n_jobs=N_JOBS),
            ExperimentalDataset("hard", n=N_DATA_ROWS,
                                difficulty=37).build(f"data/{EXP_PREFIX}_mix_hard_{N_DATA_ROWS}.hdf5", n_jobs=N_JOBS),
            ExperimentalDataset("very_hard", n=N_DATA_ROWS,
                                difficulty=70).build(f"data/{EXP_PREFIX}_mix_very_hard_{N_DATA_ROWS}.hdf5",
                                                     n_jobs=N_JOBS)]
MODELS = ['early_integration', 'intermediate_integration', 'late_integration']

DATASETS[2].mc.plot_example()
def add_config_data(exp: Experiment):
    for exp_data in DATASETS:
        exp.add_data(exp_data)


def add_config_models(exp: Experiment):
    common_model_kwargs = {'opt': 'adam', 'batch_size': int(min(N_DATA_ROWS, 15000)), 'lr': 0.0075}
    for int_type in MODELS:
        mod = ExperimentalModel(MultisensoryClassifier(integration_type=int_type, **common_model_kwargs),
                                name=int_type)
        exp.add_model(mod)


if __name__ == "__main__":
    limit_gpu_memory(5000)

    exp = Experiment(name=f'{EXP_PREFIX}_experiment', n_epochs=N_EPOCHS, n_reps=N_REPS)
    add_config_data(exp)
    add_config_models(exp)
    exp.run()
