class Spring:
    """Base class. Subclasses that act on a lever define force(stretch)."""

    def __init__(self, mass):
        self.mass = mass


class LinearSpring(Spring):
    """Hooke's law spring. The force is negative while stretched, so the spring pulls back."""

    def __init__(self, stiffness, initial_length, initial_stretch, mass=0):
        super().__init__(mass)

        self.stiffness = stiffness
        self.initial_length = initial_length
        self.initial_stretch = initial_stretch

    def force(self, stretch):
        return -self.stiffness * stretch
