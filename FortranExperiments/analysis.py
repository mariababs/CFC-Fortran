import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
import glob


class Iteration:
    def __init__(self, content_line):
        content = content_line.split()
        self.iteration = int(content[0])
        self.omtol = float(content[2])
        self.psitol = float(content[3])
        base_time = " ".join(content[5:])
        date = datetime.datetime.strptime(base_time, "%c")

        self.date = date


def load_log_file(file_name):
    data = None
    with open(file_name, "r") as f:
        contents = f.read()
        print(f"{file_name=} : {len(contents)=}")
        contents = contents.split("\n")
        start_line = 0
        end_line = 0
        dt_line = 0
        for i, l in enumerate(contents):
            if "Grid Spacing" in l:
                dt_line = i
            if "finished init" in l:
                start_line = i + 1
            if "Writing out" in l:
                end_line = i

        # extract simulation dt
        dt = float(contents[dt_line].split()[-1])

        contents = contents[start_line:end_line]
        contents = [Iteration(content_line) for content_line in contents]

        iterations = [i.iteration for i in contents]
        dts = [i * dt for i in iterations]
        omtols = [i.omtol for i in contents]
        psitols = [i.psitol for i in contents]
        time_deltas = [(i.date - contents[0].date).total_seconds() for i in contents]

        data = np.array([iterations, dts, omtols, psitols, time_deltas])
        data = pd.DataFrame(
            data.T,
            columns=["iterations", "sim_times", "omtols", "psitols", "time_deltas"],
        )

    return data


def get_log_file_names():
    log_files = glob.glob("*.log")
    return log_files


def plot_simulation_results(
    simulation_log_file_names, x_axis_label="sim_times", save_as=None
):
    fig, axes = plt.subplots(2, 2)
    (ax1, ax2, ax3, ax4) = [i for r in axes for i in r]
    for log_file_name in simulation_log_file_names:
        data = load_log_file(log_file_name)

        ax = ax1
        ax.set_yscale("log")
        ax.plot(data[x_axis_label], data.omtols)
        ax.title.set_text("Omega Tolerance Levels")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("OmegaTol")

        ax = ax2
        ax.set_yscale("log")
        ax.plot(data[x_axis_label], data.psitols)
        ax.title.set_text("Psi Tolerance Levels")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("PsiTol")

        ax = ax3
        ax.plot(data.time_deltas, data.iterations)
        ax.title.set_text("Iterations vs. Runtime")
        ax.set_xlabel("Runtime")
        ax.set_ylabel("Iteration")

        ax = ax4
        ax.plot(data.time_deltas, data.sim_times)
        ax.title.set_text("Simulation Time vs. Runtime")
        ax.set_xlabel("Runtime")
        ax.set_ylabel("Simulation Time")

    ax1.legend(simulation_log_file_names)
    # ax2.legend(simulation_log_file_names)
    # ax3.legend(simulation_log_file_names)
    # ax4.legend(simulation_log_file_names)

    if save_as is None:
        plt.show()
    else:
        plt.savefig(fig, "OmegaPsiTolVsTime.png")


def main():
    log_files = get_log_file_names()
    plot_simulation_results(log_files, x_axis_label="sim_times")


if __name__ == "__main__":
    main()