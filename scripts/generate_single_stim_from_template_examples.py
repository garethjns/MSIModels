from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate
from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events, template_noisy_sine_events


def unisensory_stim_template_examples():
    """
    TwoGapStim created from templates. Uses functions to build example params.
    """

    # Example stim using and modifying template with sine events:
    # Build params
    params = template_noisy_sine_events(fs=800)
    # Build stim
    stim = TwoGapStim(params)
    # Audiodag objects for each can be accessed using .y (this will be changed to y_obs in the future)
    _ = stim.y
    stim.y.plot(show=False)
    stim.y_mask.plot(show=True)

    # Example stims with increasing events
    for n in range(8, 14):
        try:
            stim = TwoGapStim(template_sine_events(n_events=n))
            stim.y.plot(show=False)
            stim.y_mask.plot(show=True)

        # May fail due to time incompatibility
        except IncompatibleParametersException as e:
            print(e)


def multisensory_stim_template_examples():
    """
    MultiTwoGapStim created from templates. Uses MultiTwoGapTemplate class to build params.
    """

    # Unmatched
    # Build params
    params = MultiTwoGapTemplate['unmatched_async'].set_options(background_mag=0.05).build()
    # Build stim
    multi_stim_unmatched = MultiTwoGapStim(params)
    # Audiodag objects for each can be accessed using .y_obs:
    _ = multi_stim_unmatched.y_objs
    # Or the rendered signal as an array using .y
    _ = multi_stim_unmatched.y
    # Plot:
    multi_stim_unmatched.plot(show=True)

    # Matched
    params = MultiTwoGapTemplate['matched_async'].set_options(background_mag=0.05).build()
    multi_stim_matched = MultiTwoGapStim(params)
    multi_stim_matched.plot(show=True)

    # Synchronous
    params = MultiTwoGapTemplate['matched_sync'].set_options(background_mag=0.05).build()
    multi_stim_matched = MultiTwoGapStim(params)
    multi_stim_matched.plot(show=True)


if __name__ == "__main__":
    unisensory_stim_template_examples()
    multisensory_stim_template_examples()
