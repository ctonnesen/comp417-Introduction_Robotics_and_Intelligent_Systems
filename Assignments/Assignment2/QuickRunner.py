import statistics
import subprocess as sp
import numpy as np
from inverted_pendulum import main


def regular_PID():
    keys = []
    values = []
    for i in np.arange(39, 40, 3):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "True", "--kp_heuristic", "1.6", "--ki_heuristic",
             "-0.001",
             "--kd_heuristic", "0", "--print_diffs", "True", "--rounds", "1", "--generate_disturbance", "False"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        keys.append(i)
        values.append(statistics.mean(outputs))
    return keys, values


def main():
    # point_uniform()
    # point_gaussian()
    # point_changing_step()
    regular_PID()


main()
