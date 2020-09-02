import os

from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_templates import template_sync, template_unmatched, template_matched

N = 250000
N_JOBS = 28
BATCH_SIZE = 100

if __name__ == "__main__":

    fn = 'data/sample_multisensory_data_sync_med_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(template=template_sync,
                                 fs=500,
                                 n=N,
                                 batch_size=BATCH_SIZE,
                                 fn=fn,
                                 n_jobs=N_JOBS,
                                 template_kwargs={"duration": 1300,
                                                  "background_mag": 0.09,
                                                  "duration_tol": 0.5})

    fn = 'data/sample_multisensory_data_matched_med_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(template=template_matched,
                                 fs=500,
                                 n=N,
                                 batch_size=BATCH_SIZE,
                                 fn=fn,
                                 n_jobs=N_JOBS,
                                 template_kwargs={"duration": 1300,
                                                  "background_mag": 0.09,
                                                  "duration_tol": 0.5})

    fn = 'data/sample_multisensory_data_unmatched_med_250k.hdf5'
    if not os.path.exists(fn):
        MultiTwoGapStim.generate(template=template_unmatched,
                                 fs=500,
                                 n=N,
                                 batch_size=BATCH_SIZE,
                                 fn=fn,
                                 n_jobs=N_JOBS,
                                 template_kwargs={"duration": 1300,
                                                  "background_mag": 0.09,
                                                  "duration_tol": 0.5})
