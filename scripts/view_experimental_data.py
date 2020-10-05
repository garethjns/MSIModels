from msi_models.experiment.experimental_dataset import ExperimentalDataset

if __name__ == "__main__":
    exp_data = ExperimentalDataset("example_mix", n=500, difficulty=70).build("data/example_mix7.hdf5")

    exp_data.mc.plot_summary()
    exp_data.mc.plot_example()
    exp_data.mc.plot_example()
