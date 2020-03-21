

if __name__ == "__main__":
    from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events

    common_config_kwargs = {"duration": 1300,
                            "fs": 500,
                            "background_mag": 0.09,
                            "duration_tol": 0.5}

    generate_multisensory(config_left=template_noisy_sine_events(**common_config_kwargs),
                          config_right=template_noisy_sine_events(**common_config_kwargs),
                          n=2000,
                          batch_size=10,
                          fn='multisensory_data.hdf5',
                          n_jobs=-2)
