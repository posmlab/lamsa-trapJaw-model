class Mass:
    """Load mass, as seen by the motor or spring."""

    def __init__(self, mass):
        self.mass = mass


class OneDMass(Mass):
    def __init__(self, m_end, m_rod=0, EMA=1):
        mass = m_end / EMA**2 + m_rod * ((1 + 1 / EMA)**2 + 3 * (1 / EMA - 1)**2) / 12

        super().__init__(mass)
