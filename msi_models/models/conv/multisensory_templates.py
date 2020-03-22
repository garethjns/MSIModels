from msi_models.models.conv.multisensory_base import MultisensoryBase


class MultisensoryClassifier(MultisensoryBase):
    _loss = {"agg_y_dec": "categorical_crossentropy",
             "agg_y_rate": "mse"}
    _loss_weights = {"agg_y_dec": 0.5,
                     "agg_y_rate": 0.5}
    _metrics = {"agg_y_dec": ['accuracy']}


if __name__ == "__main__":
    pass
