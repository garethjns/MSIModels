from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msi_models.models.keras_sk_base import KerasSKBasel
from msi_models.stimset.channel import Channel
from msi_models.stimset.multi_channel import MultiChannel


class ExperimentalModel:
    def __init__(self, data: Union[Channel, MultiChannel], model: KerasSKBasel,
                 name: str = "unnamed_model"):
        self.name = name
        self.data = data
        self.model = model
        self.preds_train: Dict[str, np.ndarray] = None
        self.preds_train: Dict[str, np.ndarray] = None

    def fit(self):
        self.model.fit(self.data.x_train, self.data.y_train,
                       shuffle=True,
                       validation_split=0.2,
                       batch_size=2000,
                       epochs=500,
                       verbose=2)

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

    def evaluate(self):
        self.preds_train = self._predict_batches(self.data.x_train)
        self.preds_test = self._predict_batches(self.data.x_test)

    def plot_example(self,
                     show: bool = True,
                     y_layer: str = "conv_1",
                     mistake: bool = False):

        if mistake:
            mistakes = ~((self.preds_test["dec_output"][:, 1] > 0.5)
                         == (self.data.y_test["dec_output"][:, 1].astype(bool)))
            row = np.random.choice(np.where(mistakes)[0])
        else:
            row = np.random.choice(range(0, self.data.n_test))

        for k, v in self.data.y_test.items():
            print(f"True: {k}: {v[row]}")
            print(f"Preds: {k}: {self.preds_test[k][row]}")

        for v in self.data.x_test.values():
            plt.plot(v[row])

        plt.plot(self.preds_test[y_layer][row])

        if show:
            plt.show()

    def report(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.DataFrame({'rate_output': self.data.y_train["rate_output"],
                                 'preds_rate': self.preds_train["rate_output"].squeeze(),
                                 'dec_output': self.data.y_train["dec_output"][:, 1],
                                 'preds_dec': self.preds_train["dec_output"][:, 1]})

        test_df = pd.DataFrame({'rate_output': self.data.y_test["rate_output"],
                                'preds_rate': self.preds_test["rate_output"].squeeze(),
                                'dec_output': self.data.y_test["dec_output"][:, 1],
                                'preds_dec': self.preds_test["dec_output"][:, 1]})

        return train_df, test_df
