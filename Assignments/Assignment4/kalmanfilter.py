import numpy as np


class kalmanfil:
    def __init__(self, init_theta, init_theta_dot, R, Q, time_change):
        self.state_matrix = np.array([init_theta, init_theta_dot])
        self.measure_var = R
        self.process_covar = np.array([[Q, 0], [0, Q]])
        self.P = np.eye(2)
        self.F = np.array([[1, time_change], [0, 1]])

    def predict(self):
        self.state_matrix = self.F.dot(self.state_matrix)
        # G = np.array([0.5 * time_change ** 2, time_change])
        self.P = self.F.dot(self.P).dot(self.F.T) + self.process_covar

    def update(self, meas_theta):
        z = np.array([meas_theta])
        H = np.array([1, 0]).reshape((1, 2))
        r = z - H.dot(self.state_matrix)
        S = H.dot(self.P).dot(H.T) + self.measure_var
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.state_matrix = self.state_matrix + K.dot(r)
        self.P = self.P - K.dot(H).dot(self.P)
        # self.P = (np.eye(2) - K.dot(H)).dot(self.P)
        return [self.state_matrix[0], self.state_matrix[0], self.P]


# start_theta = 0
# start_theta_dot = 0
# R = 1
# Q = 1
#
# testerKal = kalmanfil(start_theta, start_theta_dot, R, Q)
# predicted = testerKal.predict(1)
# updated = testerKal.update(2, 3)
pass
