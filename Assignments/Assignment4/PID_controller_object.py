import numpy as np


class PID_controller:
    def __init__(self):
        self.prev_action = 0  # action is in torq
        self.prev_error = 0
        self.prev_integral = 0
        self.rate_limit = 0

    def get_action(self, state, image_state, controller_mode, constant_states, generate_disturbance, disturbance_1_time,
                   disturbance_2_time, random_controller=False):
        # terminal, Boolean
        # timestep, int
        # x, float, [-2.4, 2.4]
        # x_dot, float, [-inf, inf]
        # theta, float, [-pi/2, pi/2], radians
        # theta_dot, float, [-inf, inf]
        # reward, int, 0 or 1
        # image state is a (800, 400, 3) numpy image array; ignore it for assignment 2
        timestep, theta, theta_dot = state
        if random_controller:
            return np.random.uniform(-1, 1)
        if generate_disturbance == "True" and (timestep == disturbance_1_time or timestep == disturbance_2_time):
            return np.random.uniform(-1, 1)
        else:
            # In original PID, Kp = 1.6, Kd = 49, and Ki = 0.001
            kp = constant_states[0]
            ki = constant_states[1]
            kd = constant_states[2]
            action = 0
            if "P" in controller_mode:
                action += self.get_proportional(kp, theta)
            if "I" in controller_mode:
                action += self.get_integral(ki, theta)
            if "D" in controller_mode:
                action += self.get_derivative(kd, theta)
            self.prev_action = action
            self.prev_error = theta
            return action

    def get_proportional(self, kp, theta):
        return kp * theta

    def get_integral(self, ki, theta):
        # Resets the self.prev_integral if value is too high or low
        if 0.05 < self.prev_integral + theta or self.prev_integral + theta < -0.05:
            self.rate_limit += 1
            self.prev_integral = 0
            return self.prev_integral
        self.prev_integral += theta
        return self.prev_integral * ki

    def get_derivative(self, kd, theta):
        return kd * (theta - self.prev_error)
