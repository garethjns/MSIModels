from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msi_models.experiment.experimental_dataset import ExperimentalDataset
from msi_models.models.keras_sk_base import KerasSKBase


@dataclass
class ExperimentalModel:
    model: KerasSKBase
    name: str = "unnamed_model"

    def __init__(self, model: KerasSKBase,
                 name: str = "unnamed_model"):
        self.name = name
        self.model = model
        self.preds_train: Dict[str, np.ndarray] = None
        self.preds_test: Dict[str, np.ndarray] = None

        self.run_id: int
        self.results: pd.DataFrame = pd.DataFrame()

    def fit(self, data: ExperimentalDataset,
            validation_split: float = 0.4, **kwargs):
        self.model.fit(data.stimset.x_train, data.stimset.y_train,
                       shuffle=True,
                       epochs=self.model.epochs,
                       validation_split=validation_split,
                       **kwargs)

    def predict(self, data: ExperimentalDataset) -> Tuple[Dict[str, np.ndarray],
                                                          Dict[str, np.ndarray]]:
        train_preds = self._predict_batches(data.stimset.x_train)
        test_preds = self._predict_batches(data.stimset.x_test)

        return train_preds, test_preds

    def _predict_batches(self, data: Dict[str, np.ndarray],
                         chunk_size: int = 20) -> Dict[str, np.ndarray]:

        n = len(list(data.values())[0])
        n_chunks = int(np.ceil(n / chunk_size))
        split_x = {k: np.array_split(v, n_chunks) for k, v in data.items()}
        preds = []
        for chunk_i in range(n_chunks):
            x = {k: v[chunk_i] for k, v in split_x.items()}
            preds.append(self.model.predict_dict(x))

        concat_preds = {}
        for k in preds[0].keys():
            concat_preds[k] = np.concatenate([p[k] for p in preds],
                                             axis=0)

        return concat_preds

    def plot_example(self,
                     data: ExperimentalDataset,
                     show: bool = True,
                     dec_key: str = "y_dec",
                     y_layer: str = "conv_1",
                     mistake: bool = False):
        """
        Plot a random example from the test set, with output from an early conv layer.

        TODO: Inefficient, predicts for all (to find mistakes).
        TODO: Upgrade to work with subplots for multisensory would be nice.
        """

        train_preds, test_preds = self.predict(data)

        if mistake:
            mistakes = ~((train_preds[dec_key][:, 1] > 0.5)
                         == (data.stimset.y_test[dec_key][:, 1].astype(bool)))
            row = np.random.choice(np.where(mistakes)[0])
        else:
            row = np.random.choice(range(0, test_preds[dec_key].shape[0]))

        for k, v in data.stimset.y_test.items():
            if k in test_preds.keys():
                print(f"True: {k}: {v[row]}")
                print(f"Preds: {k}: {test_preds[k][row]}")

        for v in data.stimset.x_test.values():
            plt.plot(v[row])

        plt.plot(test_preds[y_layer][row])

        if show:
            plt.show()

    def report(self, data: ExperimentalDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_preds, test_preds = self.predict(data)

        train_df = pd.DataFrame({'rate_output': data.stimset.y_train["y_rate"],
                                 'preds_rate': train_preds["y_rate"].squeeze(),
                                 'dec_output': data.stimset.y_train["y_dec"][:, 1],
                                 'preds_dec': train_preds["y_dec"][:, 1]})

        test_df = pd.DataFrame({'rate_output': data.stimset.y_test["y_rate"],
                                'preds_rate': test_preds["y_rate"].squeeze(),
                                'dec_output': data.stimset.y_test["y_dec"][:, 1],
                                'preds_dec': test_preds["y_dec"][:, 1]})

        return train_df, test_df
