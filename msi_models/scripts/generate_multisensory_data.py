from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim


N = 5000


if __name__ == "__main__":
    from msi_models.stim.multi_two_gap.multi_two_gap_templates import template_sync, template_unmatched, template_matched

    MultiTwoGapStim.generate(template=template_sync,
                             fs=500,
                             n=N,
                             batch_size=4,
                             fn='multisensory_data_sync.hdf5',
                             n_jobs=-2,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})

    MultiTwoGapStim.generate(template=template_matched,
                             fs=500,
                             n=N,
                             batch_size=4,
                             fn='multisensory_data_matched.hdf5',
                             n_jobs=-2,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})

    MultiTwoGapStim.generate(template=template_unmatched,
                             fs=500,
                             n=N,
                             batch_size=4,
                             fn='multisensory_data_unmatched.hdf5',
                             n_jobs=-2,
                             template_kwargs={"duration": 1300,
                                              "background_mag": 0.09,
                                              "duration_tol": 0.5})
