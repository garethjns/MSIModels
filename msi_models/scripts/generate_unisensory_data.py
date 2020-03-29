from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events

if __name__ == "__main__":
    TwoGapStim.generate(config=template_noisy_sine_events(duration=1300,
                                                          fs=500,
                                                          background_mag=0.09,
                                                          duration_tol=0.5),
                        n=3000,
                        batch_size=5,
                        fn='unisensory_data_hard.hdf5',
                        n_jobs=-2)
