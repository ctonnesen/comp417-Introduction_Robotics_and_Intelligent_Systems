import subprocess as sp

import numpy as np


def regular_PID():
    keys = []
    values = []
    for i in np.arange(39, 40, 3):
        run_script = sp.check_output(
            ['python', 'discrete_inverted_pendulum.py', "--mode", "non-manual", "--save_picture", "True",
             "--activate_disturbance", "True", "--save_q_table", "True", "--save_v_table", "True", "--visualize",
             "False"])
        # , "--q_table", "QTable_episode63433.npy", "--v_table", "VTable_episode63433.npy"
    return keys, values


def main():
    # point_uniform()
    # point_gaussian()
    # point_changing_step()
    regular_PID()


main()
