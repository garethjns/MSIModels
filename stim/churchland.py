from signal.digital.digital_siginal import DigitalSignal

class Churchland(DigitalSignal):
    def __init__(self,
                 base_event_sequence):

        pass


if __name__ == "__main__":

    from signal.events.tonal import SineEvent
    from signal.events.noise import NoiseEvent
    from signal.events.event import CompoundEvent

    from functools import partial
    import numpy as np

    fs = 1000
    gap_1 = partial(NoiseEvent,
                    fs=fs,
                    duration=50)
    gap_2 = partial(NoiseEvent,
                    fs=fs,
                    duration=100)
    event = partial(SineEvent,
                    fs=fs,
                    duration=100)

    start = 0
    evs = []
    for i in range(5):
        ev_f = np.random.choice([gap_1, gap_2, event])
        ev = ev_f(seed=i, start=start)

        start += ev.duration
        evs.append(ev)

    sequence = CompoundEvent(evs)
    sequence.plot(show=True)