import abc

import h5py
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


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
                    input_length: int = 650):

        input_1 = layers.Input(shape=(input_length, 1), name="input_1")

        conv_1 = layers.Conv1D(filters=256,
                               kernel_size=16,
                               name="conv_1", activation='relu')(input_1)
        flatten_1 = layers.Flatten(name="flatten_1")(conv_1)
        conv_2 = layers.Conv1D(filters=128,
                               kernel_size=8,
                               name="conv_2", activation='relu')(conv_1)

        flatten_2 = layers.Flatten(name="flatten_2")(conv_2)
        drop_1 = layers.Dropout(rate=0.1)(flatten_2)
        fc_1 = layers.Dense(int(input_length / 8),
                            activation='relu',
                            name="fc_1")(drop_1)
        fc_2 = layers.Dense(int(input_length / 32),
                            activation='relu',
                            name="fc_2")(fc_1)

        rate_output = layers.Dense(1,
                                   activation='relu',
                                   name='rate_output')(fc_2)

        dec_output = layers.Dense(2,
                                  activation='softmax',
                                  name="dec_output")(fc_2)

        model = keras.Model(inputs=input_1,
                            outputs=[rate_output, dec_output, conv_1, conv_2, flatten_1, flatten_2],
                            name='test')

        opt = keras.optimizers.Adam(lr=0.005)
        model.compile(optimizer=opt,
                      loss={"dec_output": "categorical_crossentropy",
                            "rate_output": "mse"},
                      loss_weights={"dec_output": 0.2,
                                    "rate_output": 0.8},
                      metrics={"dec_output": ['accuracy']})

        self.model = model


class UnisensoryEventDetector(KerasSKModel):

    def build_model(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    with h5py.File('data/unisensory_data.hdf5', 'r') as f:
        x = f['x'][:, :, :]
        x_indicators = f['x_indicators'][:, :, :]
        y_rate = f['y_rate'][:]
        y_dec = f['y_dec'][:, :]

    print(x.shape)

    plt.plot(x[0, :, :])
    plt.plot(x_indicators[0, :])
    plt.show()

    plt.plot(x[1, :, :])
    plt.plot(x_indicators[1, :])
    plt.show()
    import numpy as np

    shuffled_idx = np.random.choice(range(x.shape[0]), replace=False, size=x.shape[0])
    train_idx = shuffled_idx[0 : int(x.shape[0] * 0.8)]
    test_idx = shuffled_idx[int(x.shape[0] * 0.8)::]

    x_train = {'input_1': x[train_idx, :, :]}
    y_train = {'rate_output': y_rate[train_idx],
               'dec_output': y_dec[train_idx, :]}
    x_test = {'input_1': x[test_idx, :, :]}
    y_test = {'rate_output': y_rate[test_idx],
              'dec_output': y_dec[test_idx, :]}

    mod = UnisensoryCalssifier()
    mod.fit(x_train, y_train,
            shuffle=True,
            validation_split=0.2,
            batch_size=2000,
            epochs=500,
            verbose=2,
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)])

    print("preds")
    preds_train = mod.predict({k: v[0:2000] for k, v in x_train.items()})
    preds_test = mod.predict(x_test)
    print(preds_train[0][0:5])
    print(preds_test[0][0:5])

    # Conv layers
    r = 3
    print(y_train['rate_output'][r])
    plt.plot(x_train['input_1'][r, :, 0])
    plt.plot(preds_train[3][r, :, :])
    plt.show()
    plt.plot(x_train['input_1'][r, :, 0])
    plt.plot(preds_train[2][r, :, :])
    plt.show()

    # Flatten layers
    print(y_train['rate_output'][r])
    plt.plot(preds_train[4][r, :])
    plt.show()
    plt.plot(preds_train[5][r, :])
    plt.show()

    print(pd.DataFrame({'rate': y_train["rate_output"], 'preds_rate': preds_train[0].numpy().squeeze(),
                        'dec': y_train["dec_output"][:, 1], 'preds_dec': preds_train[1].numpy()[:, 1]}).head())

    print(pd.DataFrame({'rate': y_test["rate_output"], 'preds_rate': preds_test[0].numpy().squeeze(),
                        'dec': y_test["dec_output"][:, 1], 'preds_dec': preds_test[1].numpy()[:, 1]}).head())
