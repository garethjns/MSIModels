import abc
from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_core.python.keras.api._v2 import keras


class KerasSKBasel(abc.ABC, BaseEstimator):
    _loss: Dict[str, str]
    _loss_weights: Dict[str, float]
    _metrics: Dict[str, str]
    model = None

    def __init__(self, opt: str = 'adam',
                 lr: float = 0.0002,
                 es_patience: int = 100,
                 es_loss: str = 'val_loss',
                 epochs: int = 1000,
                 batch_size: int = 2000):
        self.set_params(opt=opt,
                        lr=lr,
                        es_patience=es_patience,
                        es_loss=es_loss,
                        epochs=epochs,
                        batch_size=batch_size)

    @abc.abstractmethod
    def build_model(self):
        pass

    def plot_dag(self):
        keras.utils.plot_model(self.model, 'mod.png')

    def __str__(self):
        if self.model is not None:
            return self.model.summary()
        else:
            return "Empty model"

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs)

    def fit(self, *args, **kwargs):

        if self.opt.lower() == "RMSprop":
            opt = keras.optimizers.RMSprop(lr=self.lr)
        elif self.opt.lower() == "Adagrad":
            opt = keras.optimizers.Adagrad(lr=self.lr)
        else:
            opt = keras.optimizers.Adam(lr=self.lr)

        self.build_model()
        self.model.compile(optimizer=opt,
                           loss=self._loss,
                           loss_weights=self._loss_weights,
                           metrics=self._metrics)

        es = EarlyStopping(monitor=self.es_loss,
                           mode='min',
                           verbose=2,
                           patience=self.es_patience)

        self.model.fit(*args,
                       callbacks=[es], **kwargs, )

    def predict(self, x) -> List[np.ndarray]:
        """NB: Not sklearn compatible atm."""
        preds = self.model(x)
        if not isinstance(preds, list):
            preds = [preds]

        return [p.numpy() for p in preds]

    def predict_dict(self, x) -> Dict:
        return {k: v for k, v in zip(self.model.output_names, self.predict(x))}