import numpy as np

from signal.events.event import Event


class SineEvent(Event):
    """Class specifically for tonal events."""
    def __init__(self,
                 freq: int = 2000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.freq = freq

    def _generate_f(self) -> np.ndarray:
        """Generate vector for events"""
        return np.sin(np.linspace(0, 4 * np.pi * self.freq, self.duration_pts)) * 0.5 * self.mag
