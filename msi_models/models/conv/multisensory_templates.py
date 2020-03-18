from msi_models.models.conv.multisensory_base import MultisensoryBase


class MultisensoryClassifier(MultisensoryBase):
    _loss = {"y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    _loss_weights = {"y_dec": 0.5,
                     "y_rate": 0.5}
    _metrics = {"y_dec": ['accuracy']}


if __name__ == "__main__":
    pass
