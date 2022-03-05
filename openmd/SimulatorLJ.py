from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
import h5py
from Simulator import Simulator


class SimulatorLJ(Simulator):
    def __init__(
        self,
        mass,
        sim_time: float,
        time_step: float,
        initial_values,
        box_length,
        force=None,
        force_constants=[1, 1],
        integrator=None,
        periodic=True,
    ) -> None:

        self.mass = mass
        self.sim_time = sim_time
        self.time_step = time_step
        self.num_steps = int(sim_time / time_step)
        self.initial_values = initial_values
        self.initial_pos = initial_values[0]
        self.initial_velocities = initial_values[1]
        self.box_length = box_length
        if force is not None:
            self.force = force
        self.force_constants = force_constants
        if integrator is not None:
            self.integrator = integrator
        self.periodic = periodic

    def simulate(self):
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
        # calculate chunks such that the for-loop from the integretor is written here and the integrator just called
        for i in range(self.num_steps - 1):

            positions[:, :, i + 1], velocities[:, :, i + 1] = self.integrator(
                positions=positions[:, :, i],
                velocities=velocities[:, :, i],
                iteration=i,
            )

        # needs to save to disk

        return (positions, velocities)

    def integrator(self, positions, velocities, iteration):
        """Abstracted the integrator to the core algorithm because the loop happens in the war

        Args:
            positions (_type_): _description_
            velocities (_type_): _description_

        Returns:
            _type_: _description_
        """
        i = iteration

        positions = positions + self.time_step * velocities
        if self.periodic == True:
            positions[positions > self.box_length / 2] -= self.box_length
            positions[positions <= -self.box_length / 2] += self.box_length
        acc = self.calc_accel(positions[:, :], self.force_constants)
        velocities = velocities + 0.5 * acc * self.time_step
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

    def calc_accel(self, positions, constants):

        forces = self.force(
            positions=positions, constants=constants, box_length=self.box_length
        )
        acc = forces / self.mass
        return acc

    def force(self, positions, constants, box_length):
        """Standard LJ implementation with PBC. Does only return the force to introduce more modularity.

        Args:
            positions (_type_): 3D positions
            constants (_type_): sigma, epsilon constants
            box_length (_type_): length of the periodic box

        Returns:
            _type_:  array of forces
        """
        force = np.zeros((positions.shape[0], 3))
        epsilon, sigma = constants
        for i in range(len(force)):
            difference = self.calc_pairwise_distance(
                particle=i, positions=positions, box_length=box_length
            )
            if self.periodic:
                difference[difference > box_length / 2] -= box_length
                difference[difference <= -box_length / 2] += box_length
            force_t = -48 * epsilon * np.power(sigma, 12) / np.power(
                difference, 13
            ) - 24 * epsilon * np.power(sigma, 7) / np.power(difference, 7)
            force[i, :] = force_t
        return force

    def calc_pairwise_distance(self, particle, positions, box_length):
        """Standard euclidean pairwise distance. Is abstracted to enable innovation here.

        Args:
            positions (_type_): 3D positions
            box_length (_type_): box_length
        """
        particle = positions[particle, :]
        difference = positions - particle
        difference[difference > box_length / 2] -= box_length
        difference[difference <= -box_length / 2] += box_length
        difference = np.linalg.norm(difference, axis=0)
        return difference

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

    def save_to_disk():
        # variables that needed
        self.path = path
        self.title = title
        self.grid = grid
        self.results = results

        results_file = h5py.File(self.path + f"{self.title}.hdf5", "w")
        rtset = results_file.create_dataset(
            f"time", self.grid.shape, dtype="f", data=self.grid
        )
        rset = results_file.create_dataset(
            f"{self.title}", self.results.shape, dtype="f", data=self.results
        )
        results_file.close()

        # raise NotImplementedError

    def plot_results():
        # variables that needed
        self.path = path
        self.title = title
        self.X_list_name = X_list_name
        self.Y_list_name = Y_list_name
        if results_name is not None:
            self.data_name = results_name
        self.figsize = figsize
        self.dpi = dpi
        self.grid = grid
        self.results = results

        """
        Plots a graph given a list of 1D params and params name (a list of strings).
        :params title:
        :params N_list_name:
        :params N_list:
        :params params_name:
        :params params:
        :params Figsize: ( default = (15,5) )
        :params Dpi:    ( default = 300 )
        """

        plt.rcParams["figure.figsize"] = self.figsize[0], self.figsize[1]
        plt.figure()

        for i in range(len(params)):

            if len(self.grid.shape) == 1:  # shared
                plt.plot(self.grid, self.results[i], label=str(self.result_name[i]))

            if len(self.grid.shape) > 1:  # not shared
                plt.plot(self.grid, self.results[i], label=str(self.results_name[i]))

            plt.xlabel(str(self.X_list_name), fontsize=20)
            plt.ylabel(str(self.Y_list_name), fontsize=20)
            plt.legend(fontsize=15)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # plt.show()

            plt.tight_layout()
            plt.savefig(self.path + f"{self.title}.png", dpi=self.dpi)
            plt.close()
