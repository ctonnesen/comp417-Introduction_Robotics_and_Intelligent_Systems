import math
import grip
import numpy as np


class TestObject():
    def __init__(self, theta_discrete_steps, theta, timesteps, x_dot, x, theta_dot, theta_dot_discrete_steps):
        self.theta_discrete_steps = theta_discrete_steps
        self.theta_threshold_radians = math.pi / 2
        self.theta = theta
        self.timestep = timesteps
        self.x_threshold = 2.4
        self.theta_dot_threshold = 7
        self.terminal = False
        self.x_dot = x_dot
        self.x = x
        self.theta_dot = theta_dot
        self.total_reward = 0
        self.reward = 0
        self.theta_dot_discrete_steps = theta_dot_discrete_steps
        self.V_values = np.load("VTable_episode63434.npy")
        self.Q_value = np.load("QTable_episode63434.npy")

    def get_reward(self, discrete_theta):
        # small survival reward + angle reward
        continous_value = np.abs(self.from_discrete(discrete_theta, self.theta_discrete_steps,
                                                    range=[-math.pi / 2,
                                                           math.pi / 2]))
        return 0.05 + 0.95 * (self.theta_threshold_radians - continous_value
                              ) / self.theta_threshold_radians

    def to_discrete(self, value, steps, range):
        value = np.clip(value, range[0], range[1])  # Threshold it
        value = (value - range[0]) / (range[1] - range[0])  # normalize to [0, 1]
        value = int(value * steps * 0.99999)  # ensure it cannot be exactly steps
        return value

    def from_discrete(self, discrete_value, steps, range):
        value = (
                        discrete_value + 0.5) / steps  # on average the discrete value gets rounded down even if it was 19.99 -> 19 so we use +0.5 as more accurate
        value = value * (range[1] - range[0]) + range[0]
        return value

    def get_continuous_values(self):
        return (self.terminal, self.timestep, self.x,
                self.x_dot, self.theta, self.theta_dot, self.reward)

    def get_discrete_values(self):
        discrete_theta = self.to_discrete(self.theta, self.theta_discrete_steps, range=[-math.pi / 2, math.pi / 2])
        discrete_theta_dot = self.to_discrete(self.theta_dot, self.theta_dot_discrete_steps,
                                              range=[-self.theta_dot_threshold, self.theta_dot_threshold])
        return (self.terminal, self.timestep, discrete_theta, discrete_theta_dot, self.reward)


theta_discrete_steps = 40
theta = -0.2
timesteps = 0
x_dot = 1
x = 0
theta_dot = -1
theta_dot_discrete_steps = 40
discrete_theta_value = 19
discrete_theta__dot_value = 22
grip.
# test_object = TestObject(theta_discrete_steps, theta, timesteps, x_dot, x, theta_dot, theta_dot_discrete_steps)
# print(test_object.to_discrete(theta, theta_discrete_steps, range=[-math.pi / 2,
#                                                                   math.pi / 2]))
# print(test_object.get_reward(discrete_theta_value))
# print(test_object.from_discrete(discrete_theta_value, theta_discrete_steps, range=[-math.pi / 2,
#                                                                                    math.pi / 2]))
# print(test_object.from_discrete(discrete_theta__dot_value, theta_dot_discrete_steps,
#                                 range=[-test_object.theta_dot_threshold,
#                                        test_object.theta_dot_threshold]))
