from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Tuple

from tensorflow import keras
from tensorflow.keras import layers

from msi_models.models.keras_sk_base import KerasSKBase


class MultisensoryBase(KerasSKBase):
    _loss = {"left_conv_1": "mse",
             "left_conv_2": "mse",
             "left_y_rate": "mse",
             "right_conv_1": "mse",
             "right_conv_2": "mse",
             "right_y_rate": "mse",
             "y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    _loss_weights = {"left_conv_1": 0,
                     "left_conv_2": 0,
                     "left_y_rate": 0,
                     "right_y_rate": 0,
                     "right_conv_1": 0,
                     "right_conv_2": 0,
                     "agg_y_dec": 0.5,
                     "agg_y_rate": 0.5}
    _metrics = {"agg_y_dec": ['accuracy']}

    def __init__(self,
                 integration_type: str = 'early_integration',
                 opt: str = 'adam',
                 lr: float = 0.0005,
                 epochs: int = 1000,
                 batch_size: int = 2000,
                 es_patience: int = 20,
                 es_loss: str = 'val_loss',
                 input_length: int = 650,
                 conv_1_filters: int = 8, conv_1_kernel_size: int = 2, conv_1_activation: str = 'tanh',
                 max_pool_1: int = 4,
                 conv_2_filters: int = 16, conv_2_kernel_size: int = 2, conv_2_activation: str = 'tanh',
                 max_pool_2: int = 4,
                 drop_1_prop: float = 0.01,
                 drop_2_prop: float = 0.01,
                 fc_1_units_input_prop: float = 1 / 2, fc_1_activation: str = 'relu',
                 fc_2_units_input_prop: float = 1 / 4, fc_2_activation: str = 'relu'):
        super().__init__(opt=opt, lr=lr, es_patience=es_patience,
                         es_loss=es_loss, epochs=epochs, batch_size=batch_size)

        self.integration_type: str = integration_type
        self.set_params(integration_type=integration_type,
                        input_length=input_length,
                        conv_1_filters=conv_1_filters, conv_1_kernel_size=conv_1_kernel_size,
                        conv_1_activation=conv_1_activation,
                        max_pool_1=max_pool_1,
                        conv_2_filters=conv_2_filters, conv_2_kernel_size=conv_2_kernel_size,
                        conv_2_activation=conv_2_activation,
                        max_pool_2=max_pool_2,
                        drop_1_prop=drop_1_prop,
                        drop_2_prop=drop_2_prop,
                        fc_1_units_input_prop=fc_1_units_input_prop, fc_1_activation=fc_1_activation,
                        fc_2_units_input_prop=fc_2_units_input_prop, fc_2_activation=fc_2_activation)

        self.left_input_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.right_input_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.left_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.right_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.left_output_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.right_output_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.combined_layers: OrderedDictType[str, layers.Layer] = OrderedDict()
        self.combined_output_layers: OrderedDictType[str, layers.Layer] = OrderedDict()

    def _build_early_channel(self, side: str) -> Tuple[layers.Layer, layers.Layer, layers.Layer, layers.Layer]:

        x = layers.Input(shape=(self.input_length, 1), name=f"{side}_x")

        conv_1 = layers.Conv1D(filters=self.conv_1_filters, kernel_size=self.conv_1_kernel_size,
                               name=f"{side}_conv_1", activation=self.conv_1_activation)(x)
        max_pool_1 = layers.MaxPooling1D(pool_size=self.max_pool_1, name=f"{side}_max_pool_1")(conv_1)
        flatten_1 = layers.Flatten(name=f"{side}_flatten_1")(max_pool_1)

        return x, conv_1, max_pool_1, flatten_1

    def _build_intermediate_channel(self, input_layer: layers.Layer,
                                    side: str) -> Tuple[layers.Layer, layers.Layer, layers.Layer]:

        pool_size = int(self.max_pool_2 if side == 'mid' else self.max_pool_2 / 2)

        side_conv_2 = layers.Conv1D(filters=self.conv_2_filters, kernel_size=self.conv_2_kernel_size,
                                    name=f"{side}_conv_2",
                                    activation=self.conv_2_activation)(input_layer)
        side_max_pool_2 = layers.MaxPooling1D(pool_size=pool_size, name=f"{side}_max_pool_2")(side_conv_2)
        side_flatten_2 = layers.Flatten(name=f"{side}_flatten_2")(side_max_pool_2)

        return side_conv_2, side_max_pool_2, side_flatten_2

    def _build_late_channel(self, input_layer: layers.Layer,
                            side: str) -> Tuple[layers.Layer, layers.Layer]:
        n_units = self.fc_1_units_input_prop * self.input_length
        n_units = int(n_units if side == 'mid' else n_units / 2)

        fc_1 = layers.Dense(n_units, activation=self.fc_1_activation, name=f"{side}_fc_1")(input_layer)
        drop_1 = layers.Dropout(rate=self.drop_1_prop, name=f'{side}_drop_1')(fc_1)

        return fc_1, drop_1

    @staticmethod
    def _build_side_output(input_layer: layers.Layer, side: str) -> layers.Layer:
        return layers.Dense(1, activation='relu', name=f'{side}_y_rate')(input_layer)

    def _build_head(self, input_layer: layers.Layer) -> Tuple[layers.Layer, layers.Layer, layers.Layer]:

        fc_2 = layers.Dense(int(self.fc_2_units_input_prop * self.input_length),
                            activation=self.fc_2_activation, name="mid_fc_2")(input_layer)
        drop_2 = layers.Dropout(rate=self.drop_2_prop, name=f'mid_drop_2')(fc_2)

        agg_rate_output = layers.Dense(1, activation='relu', name='agg_y_rate')(drop_2)
        agg_dec_output = layers.Dense(2, activation='softmax', name="agg_y_dec")(drop_2)

        return fc_2, agg_rate_output, agg_dec_output

    def _build_model(self) -> None:
        self.model = keras.Model(inputs=[self.left_input_layers["x"], self.right_input_layers["x"]],
                                 outputs=list(self.left_layers.values()) + list(self.right_layers.values())
                                         + list(self.left_output_layers.values()) + list(
                                     self.right_output_layers.values())
                                         + list(self.combined_layers.values()) + list(
                                     self.combined_output_layers.values()),
                                 name=self.integration_type)

    def _early_integration_model(self):
        (self.left_input_layers['x'], self.left_layers['conv_1'],
         self.left_layers['max_pool_1'], left_flatten) = self._build_early_channel(side='left')
        self.left_output_layers['y_rate'] = self._build_side_output(input_layer=left_flatten, side='left')

        (self.right_input_layers['x'], self.right_layers['conv_1'],
         self.right_layers['max_pool_1'], right_flatten) = self._build_early_channel(side='right')
        self.right_output_layers['y_rate'] = self._build_side_output(input_layer=right_flatten, side='right')

        concat_1 = layers.concatenate(inputs=([self.left_layers["max_pool_1"], self.right_layers["max_pool_1"]]),
                                      name='concat_1')

        (self.combined_layers['conv_2'],
         self.combined_layers['max_pool_2'],
         flatten_2) = self._build_intermediate_channel(input_layer=concat_1, side='mid')

        self.combined_layers['fc_1'], drop_2 = self._build_late_channel(input_layer=flatten_2, side='mid')

        (self.combined_layers['fc_2'],
         self.combined_output_layers['agg_y_rate'],
         self.combined_output_layers['agg_y_dec']) = self._build_head(input_layer=drop_2)

    def _intermediate_integration_model(self):
        (self.left_input_layers['x'], self.left_layers['conv_1'],
         self.left_layers['max_pool_1'], left_flatten) = self._build_early_channel(side='left')

        (self.right_input_layers['x'], self.right_layers['conv_1'],
         self.right_layers['max_pool_1'], right_flatten) = self._build_early_channel(side='right')

        (self.left_layers["conv_2"],
         self.left_layers["max_pool_2"],
         left_flatten_2) = self._build_intermediate_channel(input_layer=self.left_layers["max_pool_1"], side='left')
        self.left_output_layers['y_rate'] = self._build_side_output(input_layer=left_flatten_2, side='left')

        (self.right_layers["conv_2"],
         self.right_layers["max_pool_2"],
         right_flatten_2) = self._build_intermediate_channel(input_layer=self.right_layers["max_pool_1"], side='right')
        self.right_output_layers['y_rate'] = self._build_side_output(input_layer=right_flatten_2, side='right')

        concat_1 = layers.concatenate(inputs=[self.left_layers["max_pool_2"], self.right_layers["max_pool_2"]],
                                      name='concat_1')
        flatten_3 = layers.Flatten(name="flatten_3")(concat_1)

        self.combined_layers['fc_1'], drop_1 = self._build_late_channel(input_layer=flatten_3, side='mid')

        (self.combined_layers['fc_2'],
         self.combined_output_layers['agg_y_rate'],
         self.combined_output_layers['agg_y_dec']) = self._build_head(input_layer=drop_1)

    def _late_integration_model(self):
        (self.left_input_layers['x'], self.left_layers['conv_1'],
         self.left_layers['max_pool_1'], left_flatten) = self._build_early_channel(side='left')

        (self.right_input_layers['x'], self.right_layers['conv_1'],
         self.right_layers['max_pool_1'], right_flatten) = self._build_early_channel(side='right')

        (self.left_layers["conv_2"],
         self.left_layers["max_pool_2"],
         left_flatten_2) = self._build_intermediate_channel(input_layer=self.left_layers["max_pool_1"], side='left')

        (self.right_layers["conv_2"],
         self.right_layers["max_pool_2"],
         right_flatten_2) = self._build_intermediate_channel(input_layer=self.right_layers["max_pool_1"], side='right')

        self.left_layers['fc_1'], left_drop_1 = self._build_late_channel(input_layer=left_flatten_2,
                                                                         side="left")
        self.left_output_layers['y_rate'] = self._build_side_output(input_layer=left_drop_1, side='left')

        self.right_layers['fc_1'], right_drop_1 = self._build_late_channel(input_layer=right_flatten_2,
                                                                           side="right")
        self.right_output_layers['y_rate'] = self._build_side_output(input_layer=right_drop_1, side='right')

        concat_1 = layers.concatenate(inputs=[left_drop_1, right_drop_1],
                                      name='concat_1')

        (self.combined_layers['fc_2'],
         self.combined_output_layers['agg_y_rate'],
         self.combined_output_layers['agg_y_dec']) = self._build_head(input_layer=concat_1)

    def build_model(self):
        if self.integration_type == "early_integration":
            self._early_integration_model()
        elif self.integration_type == "intermediate_integration":
            self._intermediate_integration_model()
        elif self.integration_type == "late_integration":
            self._late_integration_model()

        self._build_model()


if __name__ == "__main__":
    mod = MultisensoryBase("early_integration")
    mod.build_model()
    mod.plot_dag()

    mod = MultisensoryBase("intermediate_integration")
    mod.build_model()
    mod.plot_dag()

    mod = MultisensoryBase("late_integration")
    mod.build_model()
    mod.plot_dag()
