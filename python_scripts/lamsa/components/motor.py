import warnings


class Motor:
    """Base class. Subclasses define force(t, state), where state = [position, velocity]."""

    def __init__(self, max_force, range, velocity):
        self.max_force = max_force
        self.range = range
        self.velocity = velocity

    def force(self, t, state):
        raise NotImplementedError


class LinearMotor(Motor):
    """Force falls linearly from max_force at rest to zero at the max velocity.

    With no_braking, the force never goes negative.
    """

    def __init__(self, F_motor_max, v_motor_max, range_of_motion, no_braking=True):
        if v_motor_max == 0:
            warnings.warn("v_motor_max must be nonzero, setting to Inf")
            v_motor_max = float("inf")

        super().__init__(F_motor_max, range_of_motion, v_motor_max)

        self.no_braking = no_braking

    def force(self, t, state):
        position, velocity = state

        force = self.max_force * (1 - velocity / self.velocity)

        if abs(position) > self.range:
            force = 0

        if self.no_braking:
            force = max(force, 0)

        return force
