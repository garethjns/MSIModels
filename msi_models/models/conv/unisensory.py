import tensorflow

import h5py
from tensorflow import keras
from tensorflow.keras import layers

import abc

from msi_models.models.generators.unisensory import unisensory_binary


class KerasSKModel(abc.ABC):
    @abc.abstractmethod
    def build_model(self):
        pass

    def __init__(self):
        self.model: keras.Model = None
        self.build_model()

    def plot_dag(self):
        keras.utils.plot_model(self.model, 'mod.png')

    def __str__(self):
        return self.model.summary()

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, x):
        return self.model(x)


class UnisensoryCalssifier(KerasSKModel):

    def build_model(self,
                    input_Length: int = 512):
        input = layers.Input(shape=(input_Length, 1), name="input_1")

        conv_1 = layers.Conv1D(filters=64,
                               kernel_size=256,
                               name="conv_1",
                               input_shape=(input_Length, 1))(input)

        flatten = layers.Flatten(name="flatten_1")(conv_1)
        drop_1 = layers.Dropout(rate=0.1)(flatten)
        fc_1 = layers.Dense(int(input_Length / 2),
                            activation='sigmoid',
                            name="fc_1")(drop_1)

        rate_output = layers.Dense(1,
                                   activation='relu',
                                   name='rate_output')(fc_1)

        dec_output = layers.Dense(2,
                                  activation='softmax',
                                  name="dec_output")(fc_1)

        model = keras.Model(inputs=input,
                            outputs=[rate_output, dec_output],
                            name='test')

        opt = keras.optimizers.RMSprop(lr=0.0001)
        model.compile(optimizer=opt,
                      loss={"dec_output": "categorical_crossentropy",
                            "rate_output": "mse"},
                      metrics={"dec_output": ['accuracy']})

        self.model = model


class UnisensoryEventDetector(KerasSKModel):

    def build_model(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with h5py.File('data/unisensory_data.hdf5', 'r') as f:
        x = f['x'][:, :, :]
        y_rate = f['y_rate'][:]
        y_dec = f['y_dec'][:, :]

    train_idx = range(0, 700)
    test_idx = range(701, 1000)

    x_train = {'input_1': x[train_idx, :, :]}
    y_train = {'rate_output': y_rate[train_idx],
               'dec_output': y_dec[train_idx, :]}
    x_test = {'input_1': x[test_idx, :, :]}
    y_test = {'rate_output': y_rate[test_idx],
              'dec_output': y_dec[test_idx, :]}

    mod = UnisensoryCalssifier()
    mod.fit(x_train, y_train,
            validation_split=0.2,
            batch_size=100,
            epochs=50,
            verbose=2)

    x, y = next(unisensory_binary(n=6))
    print("x")
    print(x)

    print("y")
    print(y)

    print("preds")
    print(mod.predict(x_test))
