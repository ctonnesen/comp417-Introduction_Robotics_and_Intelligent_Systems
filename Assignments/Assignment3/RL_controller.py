import os.path

import numpy as np


class RL_controller:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lr = args.lr
        self.Q_value = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps, 3))  # state-action values
        if args.q_table is not None:
            self.Q_value = np.load(args.q_table)
        self.V_values = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps))  # state values
        if args.v_table is not None:
            self.V_values = np.load(args.v_table)
        self.prev_a = 0  # previous action
        # Use a previous_state = None to detect the beginning of the new round e.g. if not(self.prev_s is None): ...
        self.prev_s = None  # Previous state
        self.save_v = args.save_v_table
        self.save_q = args.save_q_table

    def reset(self):
        self.prev_a = 0
        self.prev_s = None

    def get_action(self, state, disturbance, random_controller=False, episode=0):
        terminal, timestep, theta, theta_dot, reward = state

        if random_controller:
            action = np.random.randint(0, 3)  # you have three possible actions (0,1,2)
        else:
            # use Q values to take the best action at each state
            if disturbance:
                action = np.random.randint(0, 3)
            else:
                action = np.argmax(self.Q_value[theta][theta_dot])
        if not (self.prev_s is None or self.prev_s == [theta, theta_dot]):
            self.learn(theta, theta_dot, reward, episode)
        #############################################################
        self.prev_s = [theta, theta_dot]
        self.prev_a = action
        return action

    def learn(self, theta, theta_dot, reward, episode):
        self.Q_value[self.prev_s[0]][self.prev_s[1]][self.prev_a] += self.lr * (reward + self.gamma * max(
            self.Q_value[theta][theta_dot]) - self.Q_value[self.prev_s[0]][self.prev_s[1]][self.prev_a])
        self.V_values[self.prev_s[0]][self.prev_s[1]] = max(self.Q_value[self.prev_s[0]][self.prev_s[1]])
        V_file_name_csv = "VTable_episode" + str(episode) + ".csv"
        V_file_name_npy = "VTable_episode" + str(episode) + ".npy"
        Q_file_name = "QTable_episode" + str(episode) + ".npy"
        if self.save_v == "True":
            np.savetxt(V_file_name_csv, self.V_values, delimiter=",")
            np.save(V_file_name_npy, self.V_values)
            previous_file_csv = "VTable_episode" + str(episode - 2) + ".csv"
            previous_file_npy = "VTable_episode" + str(episode - 2) + ".npy"
            if os.path.exists(previous_file_csv):
                os.remove(previous_file_csv)
            if os.path.exists(previous_file_npy):
                os.remove(previous_file_npy)
        if self.save_q == "True":
            np.save(Q_file_name, self.Q_value)
            previous_file = "QTable_episode" + str(episode - 2) + ".npy"
            if os.path.exists(previous_file):
                os.remove(previous_file)
                # os.system(f"recycle -f {previous_file}")
