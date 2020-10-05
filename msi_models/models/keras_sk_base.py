import abc
import os
import pathlib
import pickle
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


class KerasSKBase(abc.ABC, BaseEstimator):
    _loss: Dict[str, str]
    loss_weights: Dict[str, float]
    _metrics: Dict[str, str]
    model = None

    opt: str
    lr: float
    es_patience: int
    es_loss: str
    epochs: int
    batch_size: int
    tb_on: bool

    def __init__(self, opt: str = 'adam', lr: float = 0.0002, es_patience: int = 100,
                 es_loss: str = 'val_loss', epochs: int = 1000, batch_size: int = 2000, tb_on: bool = False):
        self.set_params(opt=opt, lr=lr, es_patience=es_patience, es_loss=es_loss, epochs=epochs, batch_size=batch_size,
                        tb_on=tb_on)

    @abc.abstractmethod
    def build_model(self):
        pass

    def plot_dag(self, path: str = '') -> None:
        if self.model is None:
            self.build_model()
        try:
            keras.utils.plot_model(self.model, os.path.join(path, f"{self.integration_type}_mod.png"),
                                   show_shapes=True)
        except (ImportError, AssertionError) as e:
            print(f"Failed to plot dag due to {e}")

    def __str__(self) -> str:
        if self.model is not None:
            return self.model.summary()
        else:
            return "Empty model"

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs)

    def fit(self, *args, **kwargs) -> None:

        if self.opt.lower() == "RMSprop":
            opt = keras.optimizers.RMSprop(lr=self.lr)
        elif self.opt.lower() == "Adagrad":
            opt = keras.optimizers.Adagrad(lr=self.lr)
        else:
            opt = keras.optimizers.Adam(lr=self.lr)

        self.build_model()
        self.model.compile(optimizer=opt, loss=self._loss, loss_weights=self.loss_weights, metrics=self._metrics)

        es = EarlyStopping(monitor=self.es_loss, mode='min', verbose=2, patience=self.es_patience)
        self.model.fit(*args, **kwargs, batch_size=self.batch_size,
                       callbacks=[es] + [TensorBoard(histogram_freq=5)] if self.tb_on else [])

        tf.keras.backend.clear_session()

    def predict(self, x) -> List[np.ndarray]:
        """NB: Not sklearn compatible atm."""
        preds = self.model(x)
        if not isinstance(preds, list):
            preds = [preds]

        p_numpy = [p.numpy() for p in preds]

        tf.keras.backend.clear_session()

        return p_numpy

    def predict_dict(self, x) -> Dict:
        return {k: v for k, v in zip(self.model.output_names, self.predict(x))}

    def save(self, path: str) -> None:
        pathlib.Path(path).mkdir(exist_ok=True)

        if self.model is not None:
            self.model.save_weights(os.path.join(path, "model_weights"))

        self.clear_tf()

        pickle.dump(self, open(os.path.join(path, "model.pkl"), 'wb'))

    def clear_tf(self):
        """Clear everything tf related"""
        self.model = None
        attrs = [a for a in dir(self) if not a.startswith('_')]
        for att in attrs:
            if isinstance(getattr(self, att), OrderedDict):
                setattr(self, att, OrderedDict())
        tf.keras.backend.clear_session()

    @classmethod
    def load(cls, path: str) -> "KerasSKBase":
        model_weights_path = os.path.join(path, "model_weights")
        pkl_path = os.path.join(path, "model.pkl")

        new_obj = pickle.load(open(pkl_path, 'rb'))

        if os.path.exists(f"{model_weights_path}.index"):
            new_obj.build_model()
            new_obj.model.load_weights(model_weights_path)

        return new_obj
