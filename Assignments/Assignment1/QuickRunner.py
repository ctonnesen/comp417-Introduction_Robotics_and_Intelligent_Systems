import subprocess as sp


def point_uniform():
    extProc = sp.Popen(
        ['python', 'rrt_planner_point_robot.py', "--rrt_sampling_policy", "uniform",
         "--world",
         "shot.png", "--step_size", "1"])
    not_done = 1
    while not_done:
        if sp.Popen.poll(extProc) is not None:
            not_done = 0
    sp.Popen.terminate(extProc)


def point_gaussian():
    print("\n \n \nGaussian")
    for i in range(10):
        extProc = sp.Popen(
            ['python', 'rrt_planner_point_robot.py', "--seed", f"{i}", "--rrt_sampling_policy", "gaussian",
             "--world",
             "simple.png"])
        not_done = 1
        while not_done:
            if sp.Popen.poll(extProc) is not None:
                not_done = 0
        sp.Popen.terminate(extProc)


def point_changing_step():
    for i in range(1, 11):
        for j in range(10):
            extProc = sp.Popen(
                ['python', 'rrt_planner_point_robot.py', "--rrt_sampling_policy", "gaussian", "--seed", f"{j}",
                 "--mute_building", "1", "--step_size", f"{i * 20}"])
            not_done = 1
            while not_done:
                if sp.Popen.poll(extProc) is not None:
                    not_done = 0
            sp.Popen.terminate(extProc)
        print(f"End of step size: {i * 20} \n \n")


def line_changing_size():
    for i in range(1, 11):
        for j in range(0, 10):
            extProc = sp.Popen(
                ['python', 'rrt_planner_line_robot.py', "--rrt_sampling_policy", "uniform", "--seed", f"{j}",
                 "--mute_building", "1", "--robot_length", f"{5 * i}"])
            not_done = 1
            while not_done:
                if sp.Popen.poll(extProc) is not None:
                    not_done = 0
            sp.Popen.terminate(extProc)
        print(f"End of robot size: {i * 5} \n \n")


def main():
    point_uniform()
    # point_gaussian()
    # point_changing_step()
    # line_changing_size()


main()
