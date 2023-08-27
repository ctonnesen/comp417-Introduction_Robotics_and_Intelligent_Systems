import multiprocessing
import os.path
import statistics
import subprocess as sp
import numpy as np
import math


def QvsRTheta():
    upper_limit = 40
    value = np.zeros((upper_limit, upper_limit))
    for i in np.arange(1, upper_limit + 1, 1):
        for j in np.arange(1, upper_limit + 1, 1):
            run_script = sp.check_output(
                ['python', 'inverted_pendulum.py', "--mode", "non-manual",
                 "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
                 "0.001",
                 "--kd_heuristic", "49", "--print_diffs", "regTheta", "--rounds", "3", "--generate_disturbance",
                 "False",
                 "--save_differences", "False", "--kalman_q", f"{i}", "--kalman_r", f"{j}", "--csv_output", "False"])
            outputs = list(map(float, run_script.decode("utf-8").replace(
                "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
            value[i - 1][j - 1] = statistics.mean(outputs)
            save_string = f"Qto{upper_limit}Rto{upper_limit}.csv"
            np.savetxt(save_string, value, delimiter=",")


def ChangeOverallTimeStep():
    upper_limit = 0.1
    step = 0.005
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadotdiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "1", "--kalman_r", "1", "--csv_output", "False", "--dt",
             f"{i}"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"OverallTimestepThetaDot{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


def ChangeKalmanTimeStep():
    upper_limit = 0.1
    step = 0.005
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadotdiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "1", "--kalman_r", "1", "--csv_output", "False",
             "--separate_kalman_dt", f"{i}"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"KalmanTimestepThetaDot{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


def multi_process_QR_child(start, upper_limit):
    value = np.zeros(upper_limit * 2 + 1)
    for j in np.arange(-upper_limit, upper_limit + 1, 1):
        if j == 0:
            continue
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadotdiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", f"{start}", "--kalman_r", f"{j}", "--csv_output", "False",
             "--current_state_modifier", f"{start}"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[upper_limit + j] = statistics.mean(outputs)
        save_string = f"Qat{start}Rto{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")


def multi_process_controller():
    QR_upper_limit = 10
    time_upper_limit = 1
    step = 0.005
    # for i in range(-QR_upper_limit, QR_upper_limit + 1):
    #     # multi_process_child(i, upper_limit)
    #     process = multiprocessing.Process(target=multi_process_QR_child, args=(i, QR_upper_limit))
    #     process.start()
    # time_dot_overall_process = multiprocessing.Process(target=multi_process_timestep_overall_theta_dot_child,
    #                                                    args=(1, 0.005))
    # time_dot_overall_process.start()
    # time_dot_kalman_process = multiprocessing.Process(target=multi_process_timestep_kalman_theta_dot_child,
    #                                                   args=(1, 0.005))
    # time_dot_kalman_process.start()
    # time_theta_overall_process = multiprocessing.Process(target=multi_process_timestep_overall_theta_child,
    #                                                      args=(1, 0.005))
    # time_theta_overall_process.start()
    # time_theta_kalman_process = multiprocessing.Process(target=multi_process_timestep_kalman_theta_child,
    #                                                     args=(1, 0.005))
    # time_theta_kalman_process.start()
    kalman = os.path.exists("KalmanTimeChangeValues.csv")
    overall = os.path.exists("OverallTimeChangeValues.csv")
    if kalman:
        os.remove("KalmanTimeChangeValues.csv")
    if overall:
        os.remove("OverallTimeChangeValues.csv")
    kal = open("KalmanTimeChangeValues.csv", "w")
    over = open("OverallTimeChangeValues.csv", "w")
    kal.close()
    over.close()
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    for Q in range(-QR_upper_limit, QR_upper_limit + 1):
        for R in range(QR_upper_limit, -QR_upper_limit - 1, -1):
            if R == 0:
                continue
            for time_change in range(0, 2):
                for dt in np.arange(-time_upper_limit, time_upper_limit + step, step):
                    if not time_change:
                        for kalman_dt in np.arange(time_upper_limit, -time_upper_limit - step, -step):
                            pool.apply_async(multi_loop_run,
                                             args=(Q, R, dt, kalman_dt, "False"))
                    else:
                        pass
                        # pool.apply_async(multi_loop_run,
                        #                  args=(Q, R, dt, 0.005, "True"))


def multi_loop_run(Q, R, dt, kalman_dt, time_bool):
    run_script = sp.check_output(
        ['python', 'inverted_pendulum.py', "--mode", "non-manual",
         "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
         "0.001",
         "--kd_heuristic", "49", "--rounds", "3", "--generate_disturbance",
         "False",
         "--save_differences", "False", "--kalman_q", f"{Q}", "--kalman_r", f"{R}", "--csv_output", "False", "--dt",
         f"{dt}", "--separate_kalman_dt", f"{kalman_dt}",
         "--overall_time_change_bool", f"{time_bool}", "--assemble_overall_arrays", "True"])


def normalRun():
    run_script = sp.check_output(
        ['python', 'inverted_pendulum.py', "--mode", "non-manual",
         "--controller_mode", "PID", "--save_picture", "True", "--kp_heuristic", "1.6", "--ki_heuristic",
         "0.001",
         "--kd_heuristic", "49", "--rounds", "1", "--generate_disturbance",
         "False",
         "--save_differences", "True", "--kalman_q", "10", "--kalman_r", "4", "--csv_output", "True", "--dt",
         f"-0.015", "--separate_kalman_dt", "0.005",
         "--overall_time_change_bool", "True", "--assemble_overall_arrays",
         "True"])
    # outputs = list(map(float, run_script.decode("utf-8").replace(
    #     "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
    # test = statistics.mean(outputs)


def multi_process_timestep_overall_theta_dot_child(upper_limit, step):
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadotdiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "10", "--kalman_r", "4", "--csv_output", "False", "--dt",
             f"{i}", "--overall_time_change_bool", "True"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"OverallTimestepThetaDot{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


def multi_process_timestep_kalman_theta_dot_child(upper_limit, step):
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadotdiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "10", "--kalman_r", "4", "--csv_output", "False",
             "--separate_kalman_dt", f"{i}", "--overall_time_change_bool", "False"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"KalmanTimestepThetaDot{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


def multi_process_timestep_overall_theta_child(upper_limit, step):
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "10", "--kalman_r", "4", "--csv_output", "False", "--dt",
             f"{i}", "--overall_time_change_bool", "True"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"OverallTimestepTheta{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


def multi_process_timestep_kalman_theta_child(upper_limit, step):
    counter = 0
    value = np.zeros(math.floor(upper_limit * 2 / step) + 1)
    for i in np.arange(-upper_limit, upper_limit + step, step):
        run_script = sp.check_output(
            ['python', 'inverted_pendulum.py', "--mode", "non-manual",
             "--controller_mode", "PID", "--save_picture", "False", "--kp_heuristic", "1.6", "--ki_heuristic",
             "0.001",
             "--kd_heuristic", "49", "--print_diffs", "truekalthetadiff", "--rounds", "3", "--generate_disturbance",
             "False",
             "--save_differences", "False", "--kalman_q", "10", "--kalman_r", "4", "--csv_output", "False",
             "--separate_kalman_dt", f"{i}", "--overall_time_change_bool", "False"])
        outputs = list(map(float, run_script.decode("utf-8").replace(
            "Backend TkAgg is interactive backend. Turning interactive mode on.\r\n", "").strip().split("\r\n")))
        value[counter] = statistics.mean(outputs)
        save_string = f"KalmanTimestepTheta{-upper_limit}To{upper_limit}.csv"
        np.savetxt(save_string, value, delimiter=",")
        counter += 1


if __name__ == '__main__':
    # point_uniform()
    # point_gaussian()
    # ChangeOverallTimeStep()
    # ChangeKalmanTimeStep()
    # QvsRTheta()
    # multi_process_controller()
    normalRun()
