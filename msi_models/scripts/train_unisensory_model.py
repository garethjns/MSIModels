"""Example training unisensory model."""

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.unisensory_templates import UnisensoryClassifier
from msi_models.stimset.channel import ChannelConfig, Channel

if __name__ == "__main__":
    chan_config = ChannelConfig(path='data/unisensory_data_hard.hdf5',
                                x_keys=['x_1', 'x_indicators'],
                                y_keys=['rate_output', 'dec_output'])

    chan = Channel(chan_config)

    chan.plot_example(show=True)
    chan.plot_example(show=True)

    exp_model = ExperimentalModel(data=chan,
                                  model=UnisensoryClassifier(opt='adam',
                                                             epochs=1000,
                                                             batch_size=2500,
                                                             lr=0.0025))
    exp_model.fit()
    exp_model.evaluate()
    exp_model.plot_example()
    train_report, test_report = exp_model.report()

    exp_model.plot_example()
    exp_model.plot_example(mistake=True)
