from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
import h5py
from source.openmd.Simulator import Simulator


class SimulatorLJ(Simulator):
    """ Simulates a system of particels within a boundary influenced exclusively by Leonard Jones potential
    
        and ultilizing velocity verlet algorithm to integrate sympletic equations of motion up to 3D.
        
        the results would be saved in a directory set by user as a .hdf file.
        
    Args:
        Simulator (_type_): _description_
    """
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
        """Ultilizes velocity verlet algorithm to integrate sympletic equations of motion up to 3D with initial conditions.

        Returns:
            tuple: 3D positions and velocities of all particles in whole simulation time. 
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

        return (positions, velocities)

    def integrator(self, positions, velocities, iteration):
        """Numerical integration by Velocity-Verlet Algorithm.

        Args:
            positions (ndarry): 3D positions of all the particles in a system, in a single time frame. 
            velocities (ndarry): 3D velocities of all the particles in a system, in a single time frame.

        Returns:
            list: A list containing the simulated positions and velocities of all particles in the form [positions, velocities].
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
        """Not implemented yet
        """
        raise NotImplementedError

    def allocate_simulation(self):

        """Pre allocating RAM for the integration.

        Returns:
            tuple: 3D positions and velocities of all particles in whole simulation time.
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
        """Calculates the acceleration of the particle exerted by the force from the Leonard Jones potential.

        Args:
            positions (ndarray): 3D positions.
            constants (tuple): sigma, epsilon constants

        Returns:
            ndarray: array of accelerations
        """

        forces = self.force(
            positions=positions, constants=constants, box_length=self.box_length
        )
        acc = forces / self.mass
        return acc

    def force(self, positions, constants, box_length):
        """Standard Loenard Jones implementation with PBC. Does only return the force to introduce more modularity.

        Args:
            positions (ndarray): 3D positions
            constants (tuple): sigma, epsilon constants
            box_length (float): length of the periodic box

        Returns:
            ndarray:  array of forces
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

        return np.mean(
            np.sum(0.5 * self.mass * np.power(velocities, 2), axis=1), axis=0
        )

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

                sigma_6_temp = (
                    sigma * sigma * sigma * sigma * sigma * sigma
                )  # = sigma^6
                # sigma_12_temp = sigma_6_temp*sigma_6_temp # = sigma^12
                difference_6_temp = (
                    difference
                    * difference
                    * difference
                    * difference
                    * difference
                    * difference
                )  # = difference^6
                # difference_12_temp = difference_6_temp*difference_6_temp # = difference^12
                sigma_over_difference = sigma_6_temp / difference_6_temp
                potential_t = (
                    4 * epsilon * sigma_over_difference * (sigma_over_difference - 1)
                )

                energy_store[i] = np.mean(potential_t)

        return energy_store

    def calc_pairwise_distance(self, particle, positions, box_length):
        """Standard euclidean pairwise distance. Can be abstracted to enable innovation here.

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
        """Saves the result of the simulation of in a .hdf file. 
        The results include positions, velocities, kinetic energies, potential energies, and total energies
        of each particle in whole simulation time.

        Args:
            positions (ndarray): 3D positions of all particles in a whole simulation time.
            velocities (ndarray): 3D velocities of all particles in a whole simulation time.
            kinetic_energies (ndarray): kinetic energies of all particles in a whole simulation time.
            potential_energies (ndarray): potential energies of all particles in a whole simulation time.
            total_energy (ndarray):total energies of all particles in a whole simulation time.
        """

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
            f"mean potential energies",
            potential_energies.shape,
            data=potential_energies,
        )
        E_set = results_file.create_dataset(
            f"total mean energies", total_energy.shape, data=total_energy
        )
        results_file.close()


