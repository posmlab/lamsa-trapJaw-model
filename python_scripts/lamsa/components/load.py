class Mass:
    """Load mass, as seen by the motor or spring."""

    def __init__(self, mass):
        self.mass = mass


class OneDMass(Mass):
    def __init__(self, m_end, m_rod=0, EMA=1):
        mass = m_end / EMA**2 + m_rod * ((1 + 1 / EMA)**2 + 3 * (1 / EMA - 1)**2) / 12

        super().__init__(mass)


class RotatingMass:
    """Lever load. The pivot sits pivot_x to the side of the spring's anchor.

    inertia is about the pivot, and theta_0 is the lever angle at latch.
    """

    def __init__(self, inertia, lever_length, pivot_x, theta_0):
        self.inertia = inertia
        self.lever_length = lever_length
        self.pivot_x = pivot_x
        self.theta_0 = theta_0
