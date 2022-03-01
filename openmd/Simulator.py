import matplotlib.pyplot as plt
import h5py


class Simulator:
    def __init__(self) -> None:
        raise NotImplementedError

    def integrator():
        raise NotImplementedError

    def apply_boundary():
        raise NotImplementedError

    def calc_pairwise_distance():
        raise NotImplementedError

    def calc_force():
        raise NotImplementedError

    def calc_accel():
        raise NotImplementedError

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

        # raise NotImplementedError
