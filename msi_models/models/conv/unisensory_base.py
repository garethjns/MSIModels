from tensorflow import keras
from tensorflow.keras import layers

from msi_models.models.keras_sk_base import KerasSKBasel
from msi_models.stimset.channel import Channel


class UnisensoryBase(KerasSKBasel):
    _loss = {"conv_1": "mse",
             "conv_2": "mse",
             "flatten_1": "mse",
             "flatten_2": "mse",
             "dec_output": "categorical_crossentropy",
             "rate_output": "mse"}
    _loss_weights = {"conv_1": 0,
                     "conv_2": 0,
                     "flatten_1": 0,
                     "flatten_2": 0,
                     "dec_output": 0.5,
                     "rate_output": 0.5}
    _metrics = {"dec_output": ['accuracy']}

    def __init__(self,
                 opt: str = 'adam',
                 lr: float = 0.0002,
                 epochs: int = 1000,
                 batch_size: int = 2000,
                 es_patience: int = 100,
                 es_loss: str = 'val_loss',
                 input_length: int = 650,
                 conv_1_filters: int = 256, conv_1_kernel_size: int = 16, conv_1_activation: str = 'relu',
                 conv_2_filters: int = 128, conv_2_kernel_size: int = 8, conv_2_activation: str = 'relu',
                 drop_1_prop: float = 0.2,
                 fc_1_units_input_prop: float = 1 / 16, fc_1_activation: str = 'relu',
                 fc_2_units_input_prop: float = 1 / 32, fc_2_activation: str = 'relu'):
        super().__init__(opt=opt,
                         lr=lr,
                         es_patience=es_patience,
                         es_loss=es_loss,
                         epochs=epochs,
                         batch_size=batch_size)

        self.set_params(input_length=input_length,
                        conv_1_filters=conv_1_filters, conv_1_kernel_size=conv_1_kernel_size,
                        conv_1_activation=conv_1_activation,
                        conv_2_filters=conv_2_filters, conv_2_kernel_size=conv_2_kernel_size,
                        conv_2_activation=conv_2_activation,
                        drop_1_prop=drop_1_prop,
                        fc_1_units_input_prop=fc_1_units_input_prop, fc_1_activation=fc_1_activation,
                        fc_2_units_input_prop=fc_2_units_input_prop, fc_2_activation=fc_2_activation)

    def build_model(self,
                    input_length: int = 650):
        x_1 = layers.Input(shape=(self.input_length, 1),
                               name="x_1")

        conv_1 = layers.Conv1D(filters=self.conv_1_filters,
                               kernel_size=self.conv_1_kernel_size,
                               name="conv_1",
                               activation=self.conv_1_activation)(x_1)
        flatten_1 = layers.Flatten(name="flatten_1")(conv_1)
        conv_2 = layers.Conv1D(filters=self.conv_2_filters,
                               kernel_size=self.conv_2_kernel_size,
                               name="conv_2",
                               activation=self.conv_2_activation)(conv_1)

        flatten_2 = layers.Flatten(name="flatten_2")(conv_2)
        drop_1 = layers.Dropout(rate=self.drop_1_prop)(flatten_2)
        fc_1 = layers.Dense(int(self.fc_1_units_input_prop * self.input_length),
                            activation=self.fc_1_activation,
                            name="fc_1")(drop_1)
        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation,
                            name="fc_2")(fc_1)

        rate_output = layers.Dense(1,
                                   activation='relu',
                                   name='rate_output')(fc_2)

        dec_output = layers.Dense(2,
                                  activation='softmax',
                                  name="dec_output")(fc_2)

        self.model = keras.Model(inputs=x_1,
                                 outputs=[rate_output, dec_output, conv_1, conv_2, flatten_1, flatten_2],
                                 name='unisensory_classifier')
