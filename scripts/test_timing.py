from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import template_sync

N = 10
N_JOBS = 1
BATCH_SIZE = 5

if __name__ == "__main__":

    fn = 'data/multisensory_data_sync_time.hdf5'

    MultiTwoGapStim.generate(template=template_sync,
                             fs=500,
                             n=N,
                             batch_size=BATCH_SIZE,
                             fn=fn,
                             n_jobs=N_JOBS,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})
