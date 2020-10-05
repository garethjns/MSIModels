import os

from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate

N = 50000
N_JOBS = 28
# Each batch randomly picks a template to generate from
BATCH_SIZE = 100

if __name__ == "__main__":

    fn = f'data/scripts_multisensory_data_mix.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['left_only'],
                                            MultiTwoGapTemplate['right_only'],
                                            MultiTwoGapTemplate['matched_sync'],
                                            MultiTwoGapTemplate['matched_async'],
                                            MultiTwoGapTemplate['unmatched_async']],
                                 fs=500,
                                 n=N,
                                 batch_size=BATCH_SIZE,
                                 fn=fn,
                                 n_jobs=N_JOBS,
                                 template_kwargs={"duration": 1300,
                                                  "background_mag": 0.3,
                                                  "duration_tol": 0.5})
