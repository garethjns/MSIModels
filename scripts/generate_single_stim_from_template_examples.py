from msi_models.exceptions.params import IncompatibleParametersException
from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_templates import (template_matched, template_sync,
                                                                   template_unmatched)
from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events, template_noisy_sine_events


def unisensory_stim_template_examples():
    """
    TwoGapStim created from templates.
    """

    # Example stim using template with sine events:
    stim = TwoGapStim(template_sine_events(cache=True))
    stim.y.plot(show=False)
    stim.y_mask.plot(show=True)

    # Example stim using and modifying template with sine events::
    stim = TwoGapStim(template_noisy_sine_events(fs=800))
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
    MultiTwoGapStim created from templates.
    """

    # Unmatched
    multi_stim_unmatched = MultiTwoGapStim(template_unmatched(cache=False))
    multi_stim_unmatched.y
    multi_stim_unmatched.plot(show=True)

    # Matched
    multi_stim_matched = MultiTwoGapStim(template_matched(cache=True))
    multi_stim_matched.y
    multi_stim_matched.plot(show=True)

    # Synchronous
    multi_stim_sync = MultiTwoGapStim(template_sync(cache=True))
    multi_stim_sync.y
    multi_stim_sync.plot(show=True)


if __name__ == "__main__":
    unisensory_stim_template_examples()
    multisensory_stim_template_examples()
