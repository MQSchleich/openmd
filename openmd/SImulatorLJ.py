import numpy as np
from Simulator import Simulator


class SimulatorLJ(Simlator):
    def __init__(
        self,
        mass,
        sim_time: float,
        time_step: float,
        initial_values,
        force=None,
        force_constants=[1, 1],
        integrator=None,
        boundary_conditions=True,
    ) -> None:

        self._mass = mass
        self._sim_time = sim_time
        self._time_step = time_step
        self._num_steps = sim_time / time_step
        self._initial_values = initial_values
        self._initial_pos = initial_values[0]
        self._initial_velocities = initial_values[1]
        if force is not None:
            self._force = force
        self._force_constants = force_constants
        if integrator is not None:
            self.integrator = integrator

        @property()
        def mass(self):
            return self._mass

        @mass.setter
        def mass(self, val):
            self._mass = val

        @mass.deleter
        def mass(self):
            del self._mass

        @property()
        def sim_time(self):
            return self._sim_time

        @sim_time.setter
        def sim_time(self, val):
            self._sim_time = val

        @sim_time.deleter
        def sim_time(self):
            del self._sim_time

        @property()
        def time_step(self):
            return self._time_step

        @time_step.setter
        def time_step(self, val):
            self._time_step = val

        @time_step.deleter
        def time_step(self):
            del self._time_step

        @property()
        def initial_values(self):
            return self._initial_values

        @initial_values.setter
        def initial_values(self, values):
            self._initial_values = values
            self._initial_positions = values[0]
            self._initial_velocities = values[1]

        @initial_values.deleter
        def initial_values(self):
            del self._initial_values
            del self._initial_positions
            del self._initial_velocities

        @property()
        def initial_pos(self):
            return self._initial_pos

        @initial_pos.setter
        def initial_pos(self, values):
            self._initial_pos = values

        @initial_pos.deleter
        def initial_pos(self):
            del self._initial_pos

        @property()
        def initial_velocities(self):
            return self._initial_velocities

        @initial_velocities.setter
        def initial_velocities(self, values):
            self._initial_velocities = values

        @initial_velocities.deleter
        def initial_velocities(self):
            del self._initial_velocities

        @property()
        def force(self):
            return self._force

        @force.setter
        def initial_values(self, func):
            self._force = func

        @force.deleter
        def initial_values(self):
            del self._force

    def integration_wrapper(self):
        """
        velocity verlet algorithm to integrate sympletic equations of motion up to 3D
        :param grid:
        :param time_step:
        :param initial_values:
        :param acc_func:
        :param acc_constants:
        :return:
        """
        # assert type(force_constants) == float, "force constant must be of type float but is"+str(type(float))
        positions, velocities = self.allocate_simulation()
        positions, velocities = self.integrator(
            positions=positions, velocities=velocities
        )
        return (positions, velocities)

    def integrator(self, positions, velocities):

        acc = self.calc_accel(positions[:, :, 0], self.force_constants)
        for i in range(0, self.num_steps - 1):
            velocity_pr = velocities[:, :, i] + 0.5 * acc * self.time_step
            positions[:, :, i + 1] = positions[:, :, i] + self.time_step * velocity_pr
            # TODO implement boundary
            acc = self.calc_accel(positions[:, :, i + 1], self.force_constants)
            velocities[:, :, i + 1] = velocity_pr + 0.5 * acc * self.time_step
        return [positions, velocities]

    def apply_boundary():
        raise NotImplementedError

    def allocate_simulation(self):

        """Pre allocating RAM for the integration

        Returns:
            _type_: _description_
        """

        initial_positions = self.initial_pos
        initial_velocities = self.initial_velocities
        positions = np.zeros(
            (initial_positions.shape[0], initial_positions.shape[1], self.num_steps)
        )
        velocities = np.zeros(
            (initial_velocities.shape[0], initial_velocities.shape[1], self.num_steps)
        )
        positions[:, :, 0] = initial_positions
        velocities[:, :, 0] = initial_velocities
        return (positions, velocities)

    def calc_pairwise_distance():
        raise NotImplementedError

    def _force():
        raise NotImplementedError

    def calc_accel():
        raise NotImplementedError

    def save_to_disk():
        raise NotImplementedError

    def plot_results():
        raise NotImplementedError
