import datetime
import os
import shutil
import statistics
from os import environ
from datetime import datetime

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame, sys
import numpy as np
from pygame.locals import *
import math
import PID_controller_object
import argparse
import time
import pygame.camera
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import kalmanfilter
import csv
import filelock
import portalocker


# Base engine from the following link
# https://github.com/the-lagrangian/inverted-pendulum

class InvertedPendulum(object):
    def __init__(self, windowdims, cartdims, penddims, gravity, masspole, add_noise_to_gravity_and_mass,
                 dt, action_range=[-1, 1]):

        self.action_range = action_range

        self.window_width = windowdims[0]
        self.window_height = windowdims[1]

        self.cart_width = cartdims[0]
        self.car_height = cartdims[1]
        self.pendulum_width = penddims[0]
        self.pendulum_length = penddims[1]

        self.Y_CART = 3 * self.window_height / 4
        self.reset_state()
        if add_noise_to_gravity_and_mass:
            self.gravity = gravity + np.random.uniform(-5, 5)
            self.masscart = 1.0 + np.random.uniform(-0.5, 0.5)
            self.masspole = masspole + np.random.uniform(-0.05, 0.2)
        else:
            self.gravity = gravity
            self.masscart = 1.0
            self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.dt = dt  # seconds between state updates
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * math.pi / 360
        self.x_threshold = 2.4

        self.x_conversion = self.window_width / 2 / self.x_threshold

    def reset_state(self):
        """initializes pendulum in upright state with small perturbation"""
        self.terminal = False
        self.timestep = 0

        self.x_dot = np.random.uniform(-0.03, 0.03)
        self.x = np.random.uniform(-0.01, 0.01)

        self.theta = np.random.uniform(-0.03, 0.03)
        self.theta_dot = np.random.uniform(-0.01, 0.01)
        self.score = 0
        self.reward = 0

    def get_state(self):
        return (self.terminal, self.timestep, self.x,
                self.x_dot, self.theta, self.theta_dot, self.reward)

    def set_state(self, state):
        terminal, timestep, x, x_dot, theta, theta_dot = state
        self.terminal = terminal
        self.timestep = timestep
        self.x = x_dot
        self.x_dot = x_dot
        self.theta = theta  # in radians
        self.theta_dot = theta_dot  # in radians

    def step(self, action):
        # From OpenAI CartPole
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.timestep += 1
        if action <= -1 or 1 <= action:
            action = np.clip(action, -1, 1)  # Max action -1, 1
        force = action * 10  # multiply action by 10 to scale
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * self.theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        self.x = self.x + self.dt * self.x_dot
        self.x_dot = self.x_dot + self.dt * xacc
        self.theta = self.theta + self.dt * self.theta_dot
        self.theta_dot = self.theta_dot + self.dt * thetaacc

        self.terminal = bool(
            self.x < -self.x_threshold
            or self.x > self.x_threshold
            or self.theta < -self.theta_threshold_radians
            or self.theta > self.theta_threshold_radians
        )
        # radians to degrees
        # within -+ 15
        if (self.theta * 57.2958) < 15 and (self.theta * 57.2958) > -15:
            self.score += 1
            self.reward = 1
        else:
            self.reward = 0

        return self.get_state()


class InvertedPendulumGame(object):
    def __init__(self, figure_path, windowdims=(800, 400), cartdims=(50, 10), penddims=(6.0, 150.0), refreshfreq=1000,
                 gravity=9.81, manual_action_magnitude=1, random_controller=False, max_timestep=1000,
                 noisy_actions=False, mode=None, controller_mode=None, add_noise_to_gravity_and_mass=False,
                 save_picture="False", kp_heuristic=1.6, ki_heuristic=0.01, kd_heuristic=49, print_diffs=None,
                 rounds=None, masspole=0.1, generate_disturbance="False", save_differences="False", kalman_q=1,
                 kalman_r=1, dt=0.005, separate_kalman_dot=0.005, csv_output="False", current_state_modifier=0,
                 overall_time_change_bool="True", graph_location="GraphsAndStatesCSV",
                 state_image_location="CurrentStateImages", assemble_overall_arrays="False"):

        self.overall_time_change_bool = overall_time_change_bool
        self.obs_theta_array = None
        self.true_theta_array = None
        self.true_theta_dot_array = None
        self.obs_theta_dot_array = None
        self.kalman_theta_array = None
        self.kalman_theta_dot_array = None
        self.pos_variance_array = None
        self.vel_variance_array = None
        self.theta_diff_list = None

        self.cumulative_kalman_theta_from_0 = None
        self.cumulative_kalman_theta_diff = None
        self.cumulative_kalman_theta_dot_diff = None
        self.cumulative_overall_theta_from_0 = None
        self.cumulative_overall_theta_diff = None
        self.cumulative_overall_theta_dot_diff = None

        self.save_differences = save_differences
        self.PID_controller = mode
        self.controller_mode = controller_mode
        self.max_timestep = max_timestep
        self.pendulum = InvertedPendulum(windowdims, cartdims, penddims, gravity, masspole,
                                         add_noise_to_gravity_and_mass,
                                         dt)
        self.performance_figure_path = figure_path

        self.window_width = windowdims[0]
        self.window_height = windowdims[1]

        self.cart_width = cartdims[0]
        self.car_height = cartdims[1]
        self.pendulum_width = penddims[0]
        self.pendulum_length = penddims[1]
        self.manual_action_magnitude = manual_action_magnitude
        self.random_controller = random_controller
        self.noisy_actions = noisy_actions
        self.save_picture = save_picture
        self.kp_heuristic = kp_heuristic
        self.ki_heuristic = ki_heuristic
        self.kd_heuristic = kd_heuristic
        self.print_diffs = print_diffs
        self.rounds = rounds
        self.generate_disturbance = generate_disturbance
        self.score_list = []

        self.Y_CART = self.pendulum.Y_CART
        # self.time gives time in frames
        self.timestep = 0

        pygame.init()
        self.clock = pygame.time.Clock()
        # specify number of frames / state updates per second
        self.REFRESHFREQ = refreshfreq
        self.surface = pygame.display.set_mode(windowdims, 0, 32)
        pygame.display.set_caption('Inverted Pendulum Game')
        # array specifying corners of pendulum to be drawn
        self.static_pendulum_array = np.array(
            [[-self.pendulum_width / 2, 0],
             [self.pendulum_width / 2, 0],
             [self.pendulum_width / 2, -self.pendulum_length],
             [-self.pendulum_width / 2, -self.pendulum_length]]).T
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.prev_theta = 0
        self.prev_theta_dot = 0
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        if overall_time_change_bool == "True":
            self.kalmanfilter = kalmanfilter.kalmanfil(0, 0, kalman_r, kalman_q, dt)
        else:
            self.kalmanfilter = kalmanfilter.kalmanfil(0, 0, kalman_r, kalman_q, separate_kalman_dot)
        self.separate_kalman_dot = separate_kalman_dot
        self.csv_output = csv_output
        self.graph_location = graph_location
        self.state_image_location = state_image_location
        self.current_state_modifier = current_state_modifier
        self.assemble_overall_arrays = assemble_overall_arrays

    def draw_cart(self, x, theta):
        cart = pygame.Rect(
            self.pendulum.x * self.pendulum.x_conversion + self.pendulum.window_width / 2 - self.cart_width // 2,
            self.Y_CART, self.cart_width, self.car_height)
        pygame.draw.rect(self.surface, self.RED, cart)
        pendulum_array = np.dot(self.rotation_matrix(-theta), self.static_pendulum_array)
        pendulum_array += np.array([[x * self.pendulum.x_conversion + self.pendulum.window_width / 2], [self.Y_CART]])
        pendulum = pygame.draw.polygon(self.surface, self.BLUE,
                                       ((pendulum_array[0, 0], pendulum_array[1, 0]),
                                        (pendulum_array[0, 1], pendulum_array[1, 1]),
                                        (pendulum_array[0, 2], pendulum_array[1, 2]),
                                        (pendulum_array[0, 3], pendulum_array[1, 3])))

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-1 * np.sin(theta), np.cos(theta)]])

    def render_text(self, text, point, position="center", fontsize=48):
        font = pygame.font.SysFont(None, fontsize)
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        text_rect = text_render.get_rect()
        if position == "center":
            text_rect.center = point
        elif position == "topleft":
            text_rect.topleft = point
        self.surface.blit(text_render, text_rect)

    def time_seconds(self):
        return self.timestep / float(self.REFRESHFREQ)

    def starting_page(self):
        self.surface.fill(self.WHITE)
        self.render_text("Inverted Pendulum",
                         (0.5 * self.window_width, 0.4 * self.window_height))
        self.render_text("COMP 417 Assignment 2",
                         (0.5 * self.window_width, 0.5 * self.window_height),
                         fontsize=30)
        self.render_text("Press Enter to Begin",
                         (0.5 * self.window_width, 0.7 * self.window_height),
                         fontsize=30)
        pygame.display.update()

    def save_current_state_as_image(self, path):
        im = Image.fromarray(self.surface_array)
        im.save(
            path + f"{self.state_image_location}/Q_{self.kalman_q}_R_{self.kalman_r}_Kalman_{self.separate_kalman_dot}_DT_{self.pendulum.dt}_Change_{self.overall_time_change_bool}_Diffs_{self.print_diffs}.png")

    def get_observation(self):
        image = cv.imread(
            f"{self.state_image_location}/Q_{self.kalman_q}_R_{self.kalman_r}_Kalman_{self.separate_kalman_dot}_DT_{self.pendulum.dt}_Change_{self.overall_time_change_bool}_Diffs_{self.print_diffs}.png")
        cropped_image = image[100:400, 0:800]
        gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
        ret, thresholded_image = cv.threshold(gray_image, 50, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresholded_image, 1, 2)
        cnt = contours[0]
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(thresholded_image, [box], 0, (0, 0, 255), 2)
        cv.imshow("image", thresholded_image)
        rect_width = rect[1][0]
        rect_height = rect[1][1]
        if rect_width < rect_height:
            angle = rect[2]
        else:
            angle = rect[2] - 90
        transformed_angle = math.radians(angle)
        return transformed_angle

    def game_round(self):
        self.pendulum.reset_state()
        if self.PID_controller is not None:
            self.PID_controller = PID_controller_object.PID_controller()

        action = 0
        i = 0
        disturbance_1_time = -1
        disturbance_2_time = -2
        if self.generate_disturbance == "True":
            disturbance_1_time = np.random.randint(0, 1000)
            disturbance_2_time = np.random.randint(0, 1000)
        self.true_theta_array, self.obs_theta_array, self.true_theta_dot_array, self.obs_theta_dot_array, self.kalman_theta_array, self.kalman_theta_dot_array, self.pos_variance_array, self.vel_variance_array, self.theta_diff_list = (
            [] for _ in range(9))
        for i in range(self.max_timestep):
            if self.noisy_actions and PID_controller_object is None:
                action = action + np.random.uniform(-0.1, 0.1)
            terminal, timestep, x, _, theta, _, _ = self.pendulum.step(action)
            self.timestep = timestep
            self.surface.fill(self.WHITE)
            self.draw_cart(x, theta)
            time_text = "t = {}".format(self.pendulum.score)
            self.render_text(time_text, (0.1 * self.window_width, 0.1 * self.window_height),
                             position="topleft", fontsize=40)

            pygame.display.update()
            self.clock.tick(self.REFRESHFREQ)
            self.surface_array = pygame.surfarray.array3d(self.surface)
            self.surface_array = np.transpose(self.surface_array, [1, 0, 2])
            if self.PID_controller is None:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_LEFT:
                            action = -self.manual_action_magnitude  # "Left"
                        if event.key == K_RIGHT:
                            action = self.manual_action_magnitude
                    if event.type == KEYUP:
                        if event.key == K_LEFT:
                            action = 0
                        if event.key == K_RIGHT:
                            action = 0
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()
            else:
                self.save_current_state_as_image("")
                self.kalmanfilter.predict()
                obs_theta = self.get_observation()
                obs_theta_dot = self.prev_theta_dot + (obs_theta - self.prev_theta)
                kalman_theta, kalman_theta_dot, p_matrix = self.kalmanfilter.update(obs_theta)
                self.update_lists(obs_theta, obs_theta_dot, kalman_theta, kalman_theta_dot, p_matrix)
                self.theta_diff_list.append(np.abs(theta))
                self.prev_theta = kalman_theta
                self.prev_theta_dot = kalman_theta_dot
                state = [self.timestep, kalman_theta, kalman_theta_dot]
                action = self.PID_controller.get_action(state, self.surface_array,
                                                        self.controller_mode,
                                                        [self.kp_heuristic, self.ki_heuristic, self.kd_heuristic],
                                                        self.generate_disturbance,
                                                        disturbance_1_time,
                                                        disturbance_2_time,
                                                        random_controller=self.random_controller)
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            print("Exiting ... ")
                            pygame.quit()
                            sys.exit()
            if terminal:
                self.graph_save_coordinator()
                if self.csv_output == "True":
                    self.combine_vals()
                break

        self.score_list.append(self.pendulum.score)
        if i == self.max_timestep - 1:
            self.graph_save_coordinator()
            if self.csv_output == "True":
                self.combine_vals()

    def update_lists(self, obs_theta, obs_theta_dot, kalman_theta, kalman_theta_dot, p_matrix):
        self.true_theta_array.append(self.pendulum.theta)
        self.obs_theta_array.append(obs_theta)
        self.true_theta_dot_array.append(self.pendulum.theta_dot)
        self.obs_theta_dot_array.append(obs_theta_dot)
        self.kalman_theta_array.append(kalman_theta)
        self.kalman_theta_dot_array.append(kalman_theta_dot)
        self.pos_variance_array.append(p_matrix[0][0])
        self.vel_variance_array.append(p_matrix[1][1])

    def combine_vals(self):
        file = open(f'{self.graph_location}/StateValues_at_{datetime.now().strftime("%H-%M-%S")}.csv', 'w')
        writer = csv.writer(file, lineterminator='\n')
        header = ['TrueTheta', 'KalmanTheta', 'ThetaErrorVariance', None, None, 'TrueThetaDot', 'KalmanThetaDot',
                  'ThetaDotErrorVariance']
        writer.writerow(header)
        for i in range(len(self.true_theta_array)):
            writer.writerow(
                [self.true_theta_array[i], self.kalman_theta_array[i], self.pos_variance_array[i], None, None,
                 self.true_theta_dot_array[i], self.kalman_theta_dot_array[i], self.vel_variance_array[i]])

    def graph_save_coordinator(self):
        if self.save_picture == "True":
            titles = ["TrueTheta", "KalmanTheta", "TrueVsObsVsKalmanThetaComp", "TrueThetaDotVsKalmanComp"]
            for title in titles:
                self.graph_saves(title)
        if self.save_differences == "True":
            titles = ["TrueThetaVsKalmanDiff", "TrueThetaDotVsKalmanDiff", "TrueThetaDotVsObsDiff",
                      "TrueVsObsThetaDiff"]
            for title in titles:
                self.graph_saves(title)
        if self.print_diffs is not None:
            diff_list = []
            if self.print_diffs == "regTheta":
                diff_list = self.theta_diff_list
            if self.print_diffs == "truekalthetadiff":
                for i in range(len(self.true_theta_array)):
                    diff_list.append(self.true_theta_array[i] - self.kalman_theta_array[i])
            if self.print_diffs == "truekalthetadotdiff":
                for i in range(len(self.true_theta_array)):
                    diff_list.append(self.true_theta_dot_array[i] - self.kalman_theta_dot_array[i])
            print(statistics.mean(diff_list))

    def graph_saves(self, title):
        new_list = []
        if title == 'TrueTheta':
            plt.plot(np.arange(len(self.true_theta_array)), self.true_theta_array)
            plt.title("TrueTheta vs Time")
        if title == 'KalmanTheta':
            plt.plot(np.arange(len(self.kalman_theta_array)), self.kalman_theta_array)
            plt.title("KalmanTheta vs Time")
        if title == 'TrueVsObsVsKalmanThetaComp':
            plt.plot(np.arange(len(self.true_theta_array)), self.true_theta_array, label="True Theta Values",
                     linestyle="-")
            plt.plot(np.arange(len(self.true_theta_array)), self.obs_theta_array, label="Observation Theta Values",
                     linestyle="--")
            plt.plot(np.arange(len(self.true_theta_array)), self.kalman_theta_array, label="Kalman Theta Values",
                     linestyle=":")
            plt.legend()
            plt.ylabel('Theta(radians)')
            plt.title("Theta Values vs Time")
        if title == 'TrueThetaDotVsKalmanComp':
            plt.plot(np.arange(len(self.true_theta_array)), self.true_theta_dot_array, label="True Theta_Dot Values",
                     linestyle="-")
            plt.plot(np.arange(len(self.true_theta_array)), self.kalman_theta_dot_array,
                     label="Kalman Theta_Dot Values", linestyle=":")
            plt.legend()
            plt.ylabel('Theta(radians per timestep)')
            plt.title("Theta_dot Values vs Time")
        if title == 'TrueThetaVsKalmanDiff':
            for i in range(len(self.true_theta_array)):
                new_list.append(self.true_theta_array[i] - self.kalman_theta_array[i])
            plt.plot(np.arange(len(self.true_theta_array)), new_list)
            plt.ylabel('Theta(radians)')
            plt.title("Kalman-True-Theta-Differences vs Time")
        if title == 'TrueThetaDotVsKalmanDiff':
            for i in range(len(self.true_theta_dot_array)):
                new_list.append(self.true_theta_dot_array[i] - self.kalman_theta_dot_array[i])
            plt.plot(np.arange(len(self.true_theta_dot_array)), new_list)
            plt.ylabel('ThetaDot(radians per timestep)')
            plt.title("Kalman-True-Theta_Dot-Differences vs Time")
        if title == 'TrueThetaDotVsObsDiff':
            for i in range(len(self.true_theta_dot_array)):
                new_list.append(self.true_theta_dot_array[i] - self.obs_theta_dot_array[i])
            plt.plot(np.arange(len(self.true_theta_dot_array)), new_list)
            plt.ylabel('ThetaDot(radians per timestep)')
            plt.title("Observ-True-Theta_Dot-Differences vs Time")
        if title == 'TrueVsObsThetaDiff':
            for i in range(len(self.true_theta_array)):
                new_list.append(self.true_theta_array[i] - self.obs_theta_array[i])
            plt.plot(np.arange(len(self.true_theta_array)), new_list)
            plt.ylabel('Theta(radians)')
            plt.title("Observ-True-Theta-Differences vs Time")
        plt.xlabel('Time')
        plt.grid()
        plt.savefig(f"{self.graph_location}/" +
                    self.performance_figure_path + f"{title}" + datetime.now().strftime("%H-%M-%S") + ".png")
        #         plt.savefig(self.performance_figure_path + "_run_" + str(len(self.score_list)) + ".png")
        plt.close()

    def end_of_round(self):
        self.surface.fill(self.WHITE)
        self.draw_cart(self.pendulum.x, self.pendulum.theta)
        self.render_text("Score: {}".format(self.pendulum.score),
                         (0.5 * self.window_width, 0.3 * self.window_height))
        self.render_text("Average Score : {}".format(np.around(np.mean(self.score_list), 3)),
                         (0.5 * self.window_width, 0.4 * self.window_height))
        self.render_text("Standard Deviation Score : {}".format(np.around(np.std(self.score_list), 3)),
                         (0.5 * self.window_width, 0.5 * self.window_height))
        self.render_text("Runs : {}".format(len(self.score_list)),
                         (0.5 * self.window_width, 0.6 * self.window_height))
        if self.PID_controller is None:
            self.render_text("(Enter to play again, ESC to exit)",
                             (0.5 * self.window_width, 0.85 * self.window_height), fontsize=30)
        pygame.display.update()
        time.sleep(2.0)

    def game(self):
        self.starting_page()
        self.folder_initialize()
        self.cumulative_kalman_theta_from_0, self.cumulative_kalman_theta_diff, self.cumulative_kalman_theta_dot_diff, self.cumulative_overall_theta_from_0, self.cumulative_overall_theta_diff, self.cumulative_overall_theta_dot_diff = (
            [] for _ in range(6))
        upper_limit = 100000
        counter = 0
        if self.rounds is not None:
            upper_limit = self.rounds
        while True and counter < upper_limit:
            if self.PID_controller is None:  # Manual mode engaged
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_RETURN:
                            self.game_round()
                            self.end_of_round()
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()
            else:  # Use the PID controller instead, ignores input expect exit
                self.game_round()
                self.end_of_round()
                self.pendulum.reset_state()
                self.add_overalls()
            counter += 1
        if self.assemble_overall_arrays == "True":
            self.add_to_master_csv()

    def folder_initialize(self):

        graph_exist = os.path.isdir(self.graph_location)
        images_exist = os.path.isdir(self.state_image_location)
        if graph_exist:
            for file in os.scandir(self.graph_location):
                os.remove(file.path)
        else:
            os.mkdir(self.graph_location)
        if images_exist:
            for file in os.scandir(self.state_image_location):
                os.remove(file.path)
        else:
            os.mkdir(self.state_image_location)
        pass

    def add_overalls(self):
        if self.overall_time_change_bool == "True":
            self.cumulative_overall_theta_from_0.extend(self.theta_diff_list)
            for i in range(len(self.true_theta_array)):
                self.cumulative_overall_theta_diff.append(self.true_theta_array[i] - self.kalman_theta_array[i])
            for i in range(len(self.true_theta_array)):
                self.cumulative_overall_theta_dot_diff.append(
                    self.true_theta_dot_array[i] - self.kalman_theta_dot_array[i])
        else:
            self.cumulative_kalman_theta_from_0.extend(self.theta_diff_list)
            for i in range(len(self.true_theta_array)):
                self.cumulative_kalman_theta_diff.append(self.true_theta_array[i] - self.kalman_theta_array[i])
            for i in range(len(self.true_theta_array)):
                self.cumulative_kalman_theta_dot_diff.append(
                    self.true_theta_dot_array[i] - self.kalman_theta_dot_array[i])

    def add_to_master_csv(self):
        if self.overall_time_change_bool == "True":
            # lock = filelock.FileLock("OverallTimeChangeValues.csv.lock")
            # # with portalocker.Lock(f'OverallTimeChangeValues.csv', 'a') as file:
            # with lock:
            with open(f'OverallTimeChangeValues.csv', 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                data_line = [self.kalman_q, self.kalman_r, self.pendulum.dt, self.separate_kalman_dot,
                             statistics.mean(self.cumulative_overall_theta_from_0),
                             statistics.mean(self.cumulative_overall_theta_diff),
                             statistics.mean(self.cumulative_overall_theta_dot_diff)]
                writer.writerow(data_line)
                # file.flush()
                # os.fsync(file.fileno())
        else:
            # with portalocker.Lock(f'KalmanTimeChangeValues.csv', 'a') as file:
            # lock = filelock.FileLock("KalmanTimeChangeValues.csv.lock")
            # with lock:
            with open(f'KalmanTimeChangeValues.csv', 'a') as file:
                writer = csv.writer(file, lineterminator='\n')
                data_line = [self.kalman_q, self.kalman_r, self.pendulum.dt, self.separate_kalman_dot,
                             statistics.mean(self.cumulative_kalman_theta_from_0),
                             statistics.mean(self.cumulative_kalman_theta_diff),
                             statistics.mean(self.cumulative_kalman_theta_dot_diff)]
                writer.writerow(data_line)
                # file.flush()
                # os.fsync(file.fileno())

        # else:
        #     lock_path = f"{self.graph_location}//KalmanTimeChangeValues.csv.lock"
        #     lock = filelock.FileLock(lock_path)
        #     with lock:
        #         with open(f'{self.graph_location}/KalmanTimeChangeValues.csv', 'a') as file:
        #             fcntl.flock(file, fcntl.LOCK_EX)
        #             writer = csv.writer(file, lineterminator='\n')
        #             data_line = [self.kalman_q, self.kalman_r, self.pendulum.dt, self.separate_kalman_dot,
        #                          statistics.mean(self.cumulative_overall_theta_from_0),
        #                          statistics.mean(self.cumulative_overall_theta_diff),
        #                          statistics.mean(self.cumulative_overall_theta_dot_diff)]
        #             writer.writerow(data_line)
        #             fcntl.flock(file, fcntl.LOCK_UN)
        #             file.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="non-manual")
    parser.add_argument('--controller_mode', type=str, default="PID")
    parser.add_argument('--random_controller', type=bool, default=False)
    parser.add_argument('--add_noise_to_gravity_and_mass', type=bool, default=True)
    parser.add_argument('--max_timestep', type=int, default=1000)
    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--manual_action_magnitude', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noisy_actions', type=bool, default=False)
    parser.add_argument('--performance_figure_path', type=str, default="")
    parser.add_argument('--save_picture', type=str, default="True")
    parser.add_argument('--kp_heuristic', type=float, default=1.6)
    parser.add_argument('--ki_heuristic', type=float, default=0.001)
    parser.add_argument('--kd_heuristic', type=float, default=49)
    parser.add_argument('--print_diffs', type=str, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--masspole', type=float, default=0.1)
    parser.add_argument('--generate_disturbance', type=str, default="False")
    parser.add_argument('--save_differences', type=str, default="False")
    parser.add_argument('--kalman_q', type=float, default=1)
    parser.add_argument('--kalman_r', type=float, default=1)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--separate_kalman_dt', type=float, default=0.005)
    parser.add_argument('--csv_output', type=str, default="True")
    parser.add_argument('--current_state_modifier', type=float, default="0")
    parser.add_argument('--overall_time_change_bool', type=str, default="True")
    parser.add_argument('--graph_location', type=str, default="GraphsAndStatesCSV")
    parser.add_argument('--state_image_location', type=str, default="CurrentStateImages")
    parser.add_argument('--assemble_overall_arrays', type=str, default="False")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    np.random.seed(args.seed)
    if args.mode == "manual":
        inv = InvertedPendulumGame(args.performance_figure_path, gravity=args.gravity,
                                   manual_action_magnitude=args.manual_action_magnitude,
                                   random_controller=args.random_controller, max_timestep=args.max_timestep,
                                   noisy_actions=args.noisy_actions, mode=None,
                                   add_noise_to_gravity_and_mass=args.add_noise_to_gravity_and_mass,
                                   save_picture=args.save_picture, save_differences=args.save_differences)
    else:
        inv = InvertedPendulumGame(args.performance_figure_path, mode=PID_controller_object.PID_controller(),
                                   controller_mode=args.controller_mode, gravity=args.gravity,
                                   manual_action_magnitude=args.manual_action_magnitude,
                                   random_controller=args.random_controller, max_timestep=args.max_timestep,
                                   noisy_actions=args.noisy_actions,
                                   add_noise_to_gravity_and_mass=args.add_noise_to_gravity_and_mass,
                                   save_picture=args.save_picture, kp_heuristic=args.kp_heuristic,
                                   ki_heuristic=args.ki_heuristic, kd_heuristic=args.kd_heuristic,
                                   print_diffs=args.print_diffs, rounds=args.rounds, masspole=args.masspole,
                                   generate_disturbance=args.generate_disturbance,
                                   save_differences=args.save_differences,
                                   kalman_q=args.kalman_q, kalman_r=args.kalman_r, dt=args.dt,
                                   separate_kalman_dot=args.separate_kalman_dt,
                                   csv_output=args.csv_output, current_state_modifier=args.current_state_modifier,
                                   overall_time_change_bool=args.overall_time_change_bool,
                                   graph_location=args.graph_location,
                                   state_image_location=args.state_image_location,
                                   assemble_overall_arrays=args.assemble_overall_arrays)
    inv.game()


if __name__ == '__main__':
    main()
