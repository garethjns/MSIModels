from msi_models.experiment.experimental_dataset import ExperimentalDataset

if __name__ == "__main__":
    exp_data = ExperimentalDataset("example_easy", n=500, difficulty=15).build("data/example_mix_easy.hdf5")

    exp_data.mc.plot_summary()
    exp_data.mc.plot_example()
    exp_data.mc.plot_example()
