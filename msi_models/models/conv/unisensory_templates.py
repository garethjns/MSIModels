import h5py

from msi_models.models.conv.unisensory_base import UnisensoryBase


class UnisensoryClassifier(UnisensoryBase):
    _loss = {"y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    loss_weights = {"y_dec": 0.5,
                    "y_rate": 0.5}
    _metrics = {"y_dec": ['accuracy']}


class UnisensoryEventDetector(UnisensoryBase):
    _loss = {"conv_1": "mse",
             "conv_2": "mse",
             "y_dec": "categorical_crossentropy",
             "y_rate": "mse"}
    loss_weights = {"conv_1": 0,
                    "conv_2": 0,
                    "y_dec": 0.5,
                    "y_rate": 0.5}
    _metrics = {"y_dec": ['accuracy']}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    with h5py.File('scripts/data/sample_unisensory_data_hard.hdf5', 'r') as f:
        x = f['x'][:, :, :]
        x_indicators = f['x_indicators'][:, :, :]
        y_rate = f['y_rate'][:]
        y_dec = f['y_dec'][:, :]

    shuffled_idx = np.random.choice(range(x.shape[0]), replace=False, size=x.shape[0])
    train_idx = shuffled_idx[0: int(x.shape[0] * 0.8)]
    test_idx = shuffled_idx[int(x.shape[0] * 0.8)::]

    x_train = {'input_1': x[train_idx, :, :]}
    y_train = {'y_rate': y_rate[train_idx],
               'y_dec': y_dec[train_idx, :]}
    x_test = {'input_1': x[test_idx, :, :]}
    y_test = {'y_rate': y_rate[test_idx],
              'y_dec': y_dec[test_idx, :]}

    mod = UnisensoryClassifier(opt='adam')
    mod.fit(x_train, y_train,
            shuffle=True,
            validation_split=0.2,
            batch_size=2000,
            epochs=mod.epochs,
            verbose=2)

    print("preds")
    preds_train = mod.predict({k: v[0:2000] for k, v in x_train.items()})
    preds_test = mod.predict(x_test)
    print(preds_train[0][0:5])
    print(preds_test[0][0:5])

    # Conv layers
    r = 3
    print(y_train['rate_output'][r])
    plt.plot(x_train['input_1'][r, :, 0])
    plt.plot(preds_train[3][r, :, :])
    plt.show()
    plt.plot(x_train['input_1'][r, :, 0])
    plt.plot(preds_train[2][r, :, :])
    plt.show()

    # Flatten layers
    print(y_train['rate_output'][r])
    plt.plot(preds_train[4][r, :])
    plt.show()
    plt.plot(preds_train[5][r, :])
    plt.show()

    print(pd.DataFrame({'rate': y_train["rate_output"], 'preds_rate': preds_train[0].numpy().squeeze(),
                        'dec': y_train["dec_output"][:, 1], 'preds_dec': preds_train[1].numpy()[:, 1]}).head())

    print(pd.DataFrame({'rate': y_test["rate_output"], 'preds_rate': preds_test[0].numpy().squeeze(),
                        'dec': y_test["dec_output"][:, 1], 'preds_dec': preds_test[1].numpy()[:, 1]}).head())

    # Plot Mistake
    mistakes = ~((preds_test[1].numpy()[:, 1] > 0.5) == y_test["dec_output"][:, 1].astype(bool))
