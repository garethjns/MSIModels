"""Example running repeated model + data experiment, and logging, using ExperimentalRun"""
from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.experiment.experimental_run import ExperimentalRun
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory

N_REPS = 5
N_EPOCHS = 2
N_DATA_ROWS = 10000

if __name__ == "__main__":
    limit_gpu_memory(5000)

    # Prepare data (data doesn't need to have been pre-generated)
    data = ExperimentalDataset("scripts_example_easy",
                               n=N_DATA_ROWS, difficulty=35).build("data/scripts_example_mix_hard.hdf5")

    # Prepare model
    mod = ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                   opt='adam', batch_size=2000, lr=0.01),
                            name='scripts_example_model')

    # Prepare run
    exp_run = ExperimentalRun(name=f"scripts_example_run", model=mod, data=data,
                              n_reps=N_REPS, n_epochs=N_EPOCHS)

    # Run
    exp_run.run()

    # Evaluate
    exp_run.evaluate()

    # View results
    exp_run.log_run(to='example_run_summary')
    exp_run.log_summary(to='example_run')
    exp_run.results.plot_aggregated_results()
    print(exp_run.results.curves_agg)
