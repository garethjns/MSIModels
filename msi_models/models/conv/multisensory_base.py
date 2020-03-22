from tensorflow import keras
from tensorflow.keras import layers

from msi_models.models.keras_sk_base import KerasSKBase


class MultisensoryBase(KerasSKBase):
    _loss = {"left_conv_1": "mse",
             "left_flatten_1": "mse",
             "left_rate_output": "mse",
             "right_conv_1": "mse",
             "right_flatten_1": "mse",
             "right_rate_output": "mse",
             "y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    _loss_weights = {"left_conv_1": 0,
                     "left_flatten_1": 0,
                     "left_rate_output": 0,
                     "right_rate_output": 0,
                     "right_conv_1": 0,
                     "right_flatten_1": 0,
                     "agg_y_dec": 0.5,
                     "agg_y_rate": 0.5}
    _metrics = {"agg_y_dec": ['accuracy']}

    def __init__(self,
                 integration_type: str = 'early_integration',
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
                 drop_2_prop: float = 0.2,
                 fc_1_units_input_prop: float = 1 / 16, fc_1_activation: str = 'relu',
                 fc_2_units_input_prop: float = 1 / 32, fc_2_activation: str = 'relu'):
        super().__init__(opt=opt,
                         lr=lr,
                         es_patience=es_patience,
                         es_loss=es_loss,
                         epochs=epochs,
                         batch_size=batch_size)

        self.set_params(integration_type=integration_type,
                        input_length=input_length,
                        conv_1_filters=conv_1_filters, conv_1_kernel_size=conv_1_kernel_size,
                        conv_1_activation=conv_1_activation,
                        conv_2_filters=conv_2_filters, conv_2_kernel_size=conv_2_kernel_size,
                        conv_2_activation=conv_2_activation,
                        drop_1_prop=drop_1_prop,
                        drop_2_prop=drop_2_prop,
                        fc_1_units_input_prop=fc_1_units_input_prop, fc_1_activation=fc_1_activation,
                        fc_2_units_input_prop=fc_2_units_input_prop, fc_2_activation=fc_2_activation)

    def _early_integration_model(self):
        left_x = layers.Input(shape=(self.input_length, 1),
                              name="left_x")
        left_conv_1 = layers.Conv1D(filters=self.conv_1_filters,
                                    kernel_size=self.conv_1_kernel_size,
                                    name="left_conv_1",
                                    activation=self.conv_1_activation)(left_x)
        left_flatten_1 = layers.Flatten(name="left_flatten_1")(left_conv_1)
        left_rate_output = layers.Dense(1,
                                        activation='relu',
                                        name='left_rate_output')(left_flatten_1)

        right_x = layers.Input(shape=(self.input_length, 1),
                               name="right_x")
        right_conv_1 = layers.Conv1D(filters=self.conv_1_filters,
                                     kernel_size=self.conv_1_kernel_size,
                                     name="right_conv_1",
                                     activation=self.conv_1_activation)(right_x)
        right_flatten_1 = layers.Flatten(name="right_flatten_1")(right_conv_1)
        right_rate_output = layers.Dense(1,
                                         activation='relu',
                                         name='right_rate_output')(right_flatten_1)

        concat_1 = layers.concatenate(inputs=([left_conv_1, right_conv_1]),
                                      name='concat_1')
        # TODO: Added this layer as output to check it's sensible - remove when done

        conv_2 = layers.Conv1D(filters=self.conv_2_filters,
                               kernel_size=self.conv_2_kernel_size,
                               name="conv_2",
                               activation=self.conv_2_activation)(concat_1)
        drop_1 = layers.Dropout(rate=self.drop_1_prop)(conv_2)

        fc_1 = layers.Dense(int(self.fc_1_units_input_prop * self.input_length),
                            activation=self.fc_1_activation,
                            name="fc_1")(drop_1)
        drop_2 = layers.Dropout(rate=self.drop_2_prop,
                                name='drop_2')(fc_1)
        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation,
                            name="fc_2")(drop_2)

        agg_rate_output = layers.Dense(1,
                                       activation='relu',
                                       name='agg_y_rate')(fc_2)

        agg_dec_output = layers.Dense(2,
                                      activation='softmax',
                                      name="agg_y_dec")(fc_2)

        self.model = keras.Model(inputs=[left_x, right_x],
                                 outputs=[agg_rate_output, agg_dec_output,
                                          left_conv_1, left_flatten_1,
                                          right_conv_1, right_flatten_1,
                                          left_rate_output, right_rate_output,
                                          concat_1],  # TODO: Remove this non-standard when done
                                 name='multisensory_early')

    def _intermediate_integration_model(self):
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
        left_rate_output = layers.Dense(1,
                                        activation='relu',
                                        name='left_rate_output')(left_flatten_2)
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
        right_rate_output = layers.Dense(1,
                                         activation='relu',
                                         name='right_rate_output')(right_flatten_2)
        right_drop_1 = layers.Dropout(rate=self.drop_1_prop)(right_flatten_2)

        concat_1 = layers.concatenate(inputs=[left_drop_1, right_drop_1],
                                      name='concat_1')

        fc_1 = layers.Dense(int(self.fc_1_units_input_prop * self.input_length),
                            activation=self.fc_1_activation,
                            name="fc_1")(concat_1)
        drop_2 = layers.Dropout(rate=self.drop_2_prop,
                                name='drop_2')(fc_1)
        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation,
                            name="fc_2")(drop_2)

        agg_rate_output = layers.Dense(1,
                                       activation='relu',
                                       name='agg_y_rate')(fc_2)

        agg_dec_output = layers.Dense(2,
                                      activation='softmax',
                                      name="agg_y_dec")(fc_2)

        self.model = keras.Model(inputs=[left_x, right_x],
                                 outputs=[agg_rate_output, agg_dec_output,
                                          left_conv_1, left_flatten_1,
                                          right_conv_1, right_flatten_1,
                                          left_rate_output, right_rate_output],
                                 name='multisensory_intermediate')

    def _late_integration_model(self):
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
        fc_1_left = layers.Dense(int(self.fc_1_units_input_prop * self.input_length / 2),
                                 activation=self.fc_1_activation,
                                 name="left_fc_1")(left_drop_1)
        left_rate_output = layers.Dense(1,
                                        activation='relu',
                                        name='left_rate_output')(fc_1_left)

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
        fc_1_right = layers.Dense(int(self.fc_1_units_input_prop * self.input_length / 2),
                                  activation=self.fc_1_activation,
                                  name="right_fc_1")(right_drop_1)
        right_rate_output = layers.Dense(1,
                                         activation='relu',
                                         name='right_rate_output')(fc_1_right)

        concat_1 = layers.concatenate(inputs=[fc_1_left, fc_1_right],
                                      name='concat_1')

        drop_2 = layers.Dropout(rate=self.drop_2_prop,
                                name='drop_2')(concat_1)
        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation,
                            name="fc_2")(drop_2)

        agg_rate_output = layers.Dense(1,
                                       activation='relu',
                                       name='agg_y_rate')(fc_2)

        agg_dec_output = layers.Dense(2,
                                      activation='softmax',
                                      name="agg_y_dec")(fc_2)

        self.model = keras.Model(inputs=[left_x, right_x],
                                 outputs=[agg_rate_output, agg_dec_output,
                                          left_conv_1, left_flatten_1,
                                          right_conv_1, right_flatten_1,
                                          left_rate_output, right_rate_output],
                                 name='multisensory_late')

    def build_model(self):
        if self.integration_type == "early_integration":
            self._early_integration_model()
        elif self.integration_type == "intermediate_integration":
            self._intermediate_integration_model()
        elif self.integration_type == "late_integration":
            self._late_integration_model()
