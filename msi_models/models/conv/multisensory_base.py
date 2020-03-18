from tensorflow import keras
from tensorflow.keras import layers

from msi_models.models.keras_sk_base import KerasSKBasel


class MultisensoryBase(KerasSKBasel):
    _loss = {"left_conv_1": "mse",
             "left_conv_2": "mse",
             "left_flatten_1": "mse",
             "left_flatten_2": "mse",
             "right_conv_1": "mse",
             "right_conv_2": "mse",
             "right_flatten_1": "mse",
             "right_flatten_2": "mse",
             "y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    _loss_weights = {"left_conv_1": 0,
                     "left_conv_2": 0,
                     "left_flatten_1": 0,
                     "left_flatten_2": 0,
                     "right_conv_1": 0,
                     "right_conv_2": 0,
                     "right_flatten_1": 0,
                     "right_flatten_2": 0,
                     "y_dec": 0.5,
                     "y_rate": 0.5}
    _metrics = {"y_dec": ['accuracy']}

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
        left_x = layers.Input(shape=(self.input_length, 1),
                              name="left_x")
        left_conv_1 = layers.Conv1D(filters=self.conv_1_filters,
                                    kernel_size=self.conv_1_kernel_size,
                                    name="left_conv_1",
                                    activation=self.conv_1_activation)(left_x)
        left_flatten_1 = layers.Flatten(name="left_flatten_1")(left_conv_1)
        left_conv_2 = layers.Conv1D(filters=self.conv_2_filters,
                                    kernel_size=self.conv_2_kernel_size,
                                    name="left_conv_2",
                                    activation=self.conv_2_activation)(left_conv_1)
        left_flatten_2 = layers.Flatten(name="left_flatten_2")(left_conv_2)
        left_drop_1 = layers.Dropout(rate=self.drop_1_prop)(left_flatten_2)

        right_x = layers.Input(shape=(self.input_length, 1),
                               name="right_x")
        right_conv_1 = layers.Conv1D(filters=self.conv_1_filters,
                                     kernel_size=self.conv_1_kernel_size,
                                     name="right_conv_1",
                                     activation=self.conv_1_activation)(right_x)
        right_flatten_1 = layers.Flatten(name="right_flatten_1")(right_conv_1)
        right_conv_2 = layers.Conv1D(filters=self.conv_2_filters,
                                     kernel_size=self.conv_2_kernel_size,
                                     name="right_conv_2",
                                     activation=self.conv_2_activation)(right_conv_1)
        right_flatten_2 = layers.Flatten(name="right_flatten_2")(right_conv_2)
        right_drop_1 = layers.Dropout(rate=self.drop_1_prop)(right_flatten_2)

        add_1 = layers.Add(name="add_1")([left_drop_1, right_drop_1])

        fc_1 = layers.Dense(int(self.fc_1_units_input_prop * self.input_length),
                            activation=self.fc_1_activation,
                            name="fc_1")(add_1)
        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation,
                            name="fc_2")(fc_1)

        rate_output = layers.Dense(1,
                                   activation='relu',
                                   name='y_rate')(fc_2)

        dec_output = layers.Dense(2,
                                  activation='softmax',
                                  name="y_dec")(fc_2)

        self.model = keras.Model(inputs=[left_x, right_x],
                                 outputs=[rate_output, dec_output,
                                          left_conv_1, left_conv_2, left_flatten_1, left_flatten_2,
                                          right_conv_1, right_conv_2, right_flatten_1, right_flatten_2],
                                 name='multisensory_classifier')
