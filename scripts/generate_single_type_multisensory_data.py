import os

from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate

N = 250000
N_JOBS = 29
BATCH_SIZE = 100
DIFFICULTY = 0.18

if __name__ == "__main__":

    common_kwargs = {'fs': 500, 'n': N, 'batch_size': BATCH_SIZE, 'n_jobs': N_JOBS,
                     'template_kwargs': {"duration": 1300, "background_mag": DIFFICULTY, "duration_tol": 0.5}}

    fn = 'data/scripts_multisensory_data_matched_sync_hard_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['matched_sync']], fn=fn, **common_kwargs)

    fn = 'data/scripts_multisensory_data_matched_async_hard_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['matched_async']], fn=fn, **common_kwargs)

    fn = 'data/scripts_multisensory_data_unmatched_hard_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(templates=[MultiTwoGapTemplate['unmatched_async']], fn=fn, **common_kwargs)
