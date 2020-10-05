"""Example training a model using the experimental wrappers."""

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_classifier import MultisensoryClassifier
from msi_models.tf_helpers import limit_gpu_memory

N_DATA_ROWS = 10000

if __name__ == "__main__":
    limit_gpu_memory(3000)

    # Prepare data (doesn't need to have been pre-generated)
    data = ExperimentalDataset("scripts_example_easy",
                               n=N_DATA_ROWS, difficulty=35).build("data/scripts_example_mix_hard.hdf5")

    # Prepare 3 models types
    common_model_kwargs = {'opt': 'adam', 'batch_size': 5000, 'lr': 0.007}
    early_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='early_integration',
                                                               **common_model_kwargs))
    int_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                             **common_model_kwargs))
    late_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='late_integration',
                                                              **common_model_kwargs))

    # fit each model and print some evaluation
    # for mod in [early_exp_model, int_exp_model, late_exp_model]:
    for mod in [int_exp_model]:
        # Fit
        mod.fit(data, epochs=2000)

        # Eval
        if mod.model.integration_type == 'intermediate_integration':
            # This isn't implemented for all integration_types yet.
            mod.plot_example(data, dec_key='agg_y_dec')

        psyche_fits = mod.calc_psyche_curves(data, rate_key='agg_y_rate')
        mod.plot_prop_fast(data, rate_key='agg_y_rate')
        train_report, test_report = mod.report(data)
        print(psyche_fits)
