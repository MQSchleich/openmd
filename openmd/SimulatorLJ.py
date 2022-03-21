from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
import h5py
from Simulator import Simulator
import line_profiler


class SimulatorLJ(Simulator):
    def __init__(
        self,
        path,
        title,
        mass,
        sim_time: float,
        time_step: float,
        initial_values,
        box_length,
        force=None,
        force_constants=[1, 1],
        integrator=None,
        periodic=True,
        figsize=(10, 10),
        dpi=300,
    ) -> None:
        
        self.path = path
        self.title = title
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
        self.figsize = figsize
        self.dpi = dpi

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

        # calculate energies
        kinetic_energies = self._mean_kinetic_energy(velocities)
        potential_energies = self._mean_LJ_energy(self.force_constants, positions)
        total_energies = kinetic_energies + potential_energies
        
        # needs to save to disk & plots
        """self.save_to_disk(positions, velocities, kinetic_energies, potential_energies, total_energies)
        self.plot_results("x", positions[:, 0, :], "y", positions[:, 1, :])
        self.plot_results("y", positions[:, 1, :], "z", positions[:, 2, :])
        self.plot_results("x", positions[:, 0, :], "z", positions[:, 2, :])
        for i in range(3):
            self.plot_results(
                "time",
                np.linspace(0, self.sim_time, self.num_steps),
                "velocity",
                velocities[:, i, :],
            )
        self.plot_results(
            "time",
            np.linspace(0, self.sim_time, self.num_steps),
            "kinetic energy",
            [kinetic_energies],
        )

        self.plot_results(
            "time",
            np.linspace(0, self.sim_time, self.num_steps),
            "potential energy",
            [potential_energies],
        )

        self.plot_results(
            "time",
            np.linspace(0, self.sim_time, self.num_steps),
            "total energies",
            [total_energies],
        )"""

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

    def _mean_kinetic_energy(self, velocities):
        
        return np.mean(np.sum(0.5 * self.mass * np.power(velocities, 2), axis = 1), axis=0)

    def _mean_LJ_energy(self, constants, positions):
        
        epsilon, sigma = constants
        energy_store = np.zeros(positions.shape[2])
        for i in range(positions.shape[2]):
            positions_3D = positions[:, :, i]
            energy = np.zeros((positions.shape[0], 3))
            for j in range(len(energy)):
                difference = self.calc_pairwise_distance(
                    particle=j, positions=positions, box_length=self.box_length
                )
                if self.periodic:
                    difference[difference > self.box_length * 0.5] -= self.box_length
                    difference[difference <= -self.box_length * 0.5] += self.box_length
                    
                sigma_6_temp = sigma*sigma*sigma*sigma*sigma*sigma # = sigma^6
                # sigma_12_temp = sigma_6_temp*sigma_6_temp # = sigma^12
                difference_6_temp = difference*difference*difference*difference*difference*difference # = difference^6
                # difference_12_temp = difference_6_temp*difference_6_temp # = difference^12
                sigma_over_difference = (sigma_6_temp / difference_6_temp)
                potential_t = 4*epsilon*sigma_over_difference*(sigma_over_difference - 1)

                energy_store[i] = np.mean(potential_t)

        return energy_store

    @profile
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

    def save_to_disk(
        self, positions, velocities, kinetic_energies, potential_energies, total_energy
    ):
        # variables that needed

        results_file = h5py.File(f"{self.path}/{self.title}.hdf5", "w")
        rset = results_file.create_dataset(
            f"positions", positions.shape, data=positions
        )
        vset = results_file.create_dataset(
            f"velocities", velocities.shape, data=velocities
        )
        ke_set = results_file.create_dataset(
            f"mean kinetic energies", kinetic_energies.shape, data=kinetic_energies
        )
        pe_set = results_file.create_dataset(
            f"mean potential energies", potential_energies.shape, data=potential_energies
        )
        E_set = results_file.create_dataset(
            f"total mean energies", total_energy.shape, data=total_energy
        )
        results_file.close()

    """
    def plot_results(self, xlist_name, xlist, ylist_name, ylist):
        # variables that needed
        
        Plots a graph given a list of 1D params and params name (a list of strings).
        :params title:
        :params N_list_name:
        :params N_list:
        :params params_name:
        :params params:
        :params Figsize: ( default = (15,5) )
        :params Dpi:    ( default = 300 )
        

        plt.rcParams["figure.figsize"] = self.figsize[0], self.figsize[1]
        plt.figure()

        for i in range(len(ylist)):

            if len(xlist.shape) == 1:  # shared
                plt.plot(xlist, ylist[i], label=f"particle {i+1}")

            if len(xlist.shape) > 1:  # not shared
                plt.plot(xlist[i], ylist[i], label=f"particle {i+1}")

        plt.xlabel(str(xlist_name), fontsize=20)
        plt.ylabel(str(ylist_name), fontsize=20)
        plt.legend(fontsize=5)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # plt.show()

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/{self.title}_{xlist_name}_{ylist_name}.png", dpi=self.dpi
        )
        plt.close()"""