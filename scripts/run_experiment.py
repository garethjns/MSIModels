"""(not using Experiment, yet)"""

import gc
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from msi_models.experiment.experimental_model import ExperimentalModel
from msi_models.models.conv.multisensory_templates import MultisensoryClassifier
from msi_models.stimset.channel import ChannelConfig
from msi_models.stimset.multi_channel import MultiChannelConfig, MultiChannel

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(
                                                            memory_limit=5000)])

N_REPS = 3
N_EPOCHS = 2000


def plot_aggregated_prop_fast(exp_mods: List[ExperimentalModel], rate_key: str = 'agg_y_rate', type_key: str = 'type'):
    train_pfs = []
    test_pfs = []
    for exp_mod in exp_mods:
        train_pf, test_pf = exp_mod.calc_prop_fast(mc, rate_key='agg_y_rate')
        train_pfs.append(train_pf)
        test_pfs.append(test_pf)

    typs = np.sort(train_pf[type_key].unique())
    n_subplots = len(typs)
    fig, axs = plt.subplots(ncols=n_subplots, figsize=(2.5 * n_subplots, 8))

    for ai, (ax, ty) in enumerate(zip(axs, typs)):
        for pfs, name in zip([train_pfs, test_pfs], ['train', 'test']):
            pfs_subjects = pd.concat(pfs, axis=0).reset_index(drop=False)
            pfs_subset = pfs_subjects.loc[pfs_subjects.type == ty, [rate_key, 'preds_dec']]
            pfs_gb = pfs_subset.groupby(rate_key)
            ax.errorbar(pfs_gb[rate_key].first(), pfs_gb.mean()['preds_dec'],
                        pfs_gb.std()['preds_dec'] / np.sqrt(len(pfs)), label=name)

        ax.set_ylim([0, 1])
        ax.set_xlabel('Rate, Hz', fontweight='bold')
        if ai == 0:
            ax.set_ylabel('Prop fast decision', fontweight='bold')
            ax.legend(title='Set')
    fig.suptitle(f"{exp_mods[0].model.integration_type.capitalize()} (n={len(train_pfs)}", fontweight='bold')
    fig.show()


if __name__ == "__main__":

    # Prepare data
    fn = "data/sample_multisensory_data_mix_hard_250k.hdf5"
    path = os.path.join(os.getcwd().split('msi_models')[0], fn).replace('\\', '/')

    common_kwargs = {"path": path,
                     "train_prop": 0.8,
                     "x_keys": ["x", "x_mask"],
                     "y_keys": ["y_rate", "y_dec"]}

    models = {'early': [],
              'intermediate': [],
              'late': []}

    for _ in range(N_REPS):

        left_config = ChannelConfig(key='left', **common_kwargs)
        right_config = ChannelConfig(key='right', **common_kwargs)
        multi_config = MultiChannelConfig(path=path,
                                          key='agg',
                                          y_keys=common_kwargs["y_keys"],
                                          channels=[left_config, right_config])
        mc = MultiChannel(multi_config)

        # Prepare models
        common_model_kwargs = {'opt': 'adam',
                               'batch_size': 15000,
                               'lr': 0.007}

        early_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='early_integration',
                                                                   **common_model_kwargs))
        int_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='intermediate_integration',
                                                                 **common_model_kwargs))
        late_exp_model = ExperimentalModel(MultisensoryClassifier(integration_type='late_integration',
                                                                  **common_model_kwargs))

        for mod in [early_exp_model, int_exp_model, late_exp_model]:
            # Fit
            mod.fit(mc, epochs=N_EPOCHS)
            # Eval
            # mod.plot_example(mc, dec_key='agg_y_dec')
            # mod.calc_prop_fast(mc, dec_key='agg_y_rate')
            mod.plot_prop_fast(mc, type_key='type', rate_key='agg_y_rate')
            train_report, test_report = mod.report(mc)
            gc.collect()

        models['early'].append(early_exp_model)
        models['intermediate'].append(int_exp_model)
        models['late'].append(late_exp_model)

    plot_aggregated_prop_fast(models['early'])
    gc.collect()
    plot_aggregated_prop_fast(models['intermediate'])
    gc.collect()
    plot_aggregated_prop_fast(models['late'])

    print('stop')
