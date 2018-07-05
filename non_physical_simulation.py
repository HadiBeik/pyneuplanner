import pygame
import math
import numpy as np
import time
import random
import math
import Queue
import copy
import Vector2D as vector
import Position2D as position


class simulator:
    def __init__(self):
        pygame.display.set_caption("Simulator")
        self.history = Queue.Queue(maxsize=3)
        self.initial_angle = [0, 0]
        self.state = [0, 0]
        self.state_abs = [0, 0]
        self.observation = np.array([0, 0, 0, 0], np.float)
        self.gray = (127, 127, 127)
        self.frame_duration = 0.064
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.darkBlue = (0, 0, 128)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.max_time = 5
        self.window_size = 3
        self.pink = (255, 200, 200)
        self.screen_size = np.array((800, 700))
        self.rand_sign = (-1, 1)
        self.vel_init = np.zeros(2)
        self.min_vel = np.array((25, 25))
        self.max_vel = np.array((30, 30))
        self.min_pos = np.array((0, 0))
        self.max_pos = np.array((self.screen_size[0], self.screen_size[1] - 100))
        self.pos = np.array((0, 0))
        self.pos_init = np.array((0, 0))
        self.timer = 0
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]))
        self.screen.fill(self.white)
        pygame.draw.line(self.screen, self.black, (0, self.max_pos[1]), (self.max_pos[0], self.max_pos[1]), 5)
        self.robot_pos = (self.max_pos[0] / 2, self.max_pos[1] + 50)

        pygame.display.update()
        self.out_of_bound = False

    def init_velocity(self):
        self.vel_init[0] = (self.min_vel[0] +
                            random.random() * (self.max_vel[0] - self.min_vel[0])) * random.choice(self.rand_sign)
        self.vel_init[1] = (self.min_vel[1] +
                            random.random() * (self.max_vel[1] - self.min_vel[1])) * random.choice(self.rand_sign)

    def init_position(self):
        self.pos_init[0] = self.min_pos[0] + (random.random() * (self.max_pos[0] - self.min_pos[0]))
        self.pos_init[1] = self.min_pos[1] + (random.random() * (self.max_pos[1] - self.min_pos[1]))

    def pos_prediction(self, t):
        pos_t = np.zeros(2)
        pos_t[0] = (self.vel_init[0] * t) + self.pos_init[0]
        pos_t[1] = (self.vel_init[1] * t) + self.pos_init[1]
        return pos_t

    def init(self):
        self.robot_pos = (self.max_pos[0] / 2, self.max_pos[1] + 50)
        self.screen.fill(self.white)
        self.init_velocity()
        self.init_position()
        self.timer = 0
        self.out_of_bound = False
        self.history = Queue.Queue(maxsize=3)

    def robot_delta_t(self, target):
        dist = math.sqrt(((self.robot_pos[0] - target[0]) * (self.robot_pos[0] - target[0])) + (
                (self.robot_pos[1] - target[1]) * (self.robot_pos[1] - target[1])))
        max_dist = math.sqrt(((self.robot_pos[0] - self.min_pos[0]) * (self.robot_pos[0] - self.min_pos[0])) + (
                (self.robot_pos[1] - self.min_pos[1]) * (self.robot_pos[1] - self.min_pos[1])))
        time_robot = self.max_time * dist / max_dist
        return time_robot

    def step(self, target, reset):

        frame_num = 1
        if reset:
            frame_num = self.window_size
        for i in range(frame_num):
            time.sleep(self.frame_duration)
            t = self.timer * self.frame_duration
            self.pos = self.pos_prediction(t)
            if frame_num == 1:
                self.history.get()
            self.history.put(np.array(self.pos))
            # pygame.draw.circle(self.screen, self.red, (int(self.pos[0]), int(self.pos[1])), self.window_size, 0) # ballvis
            pygame.display.update()
            self.timer += 1
        if not self.min_pos[0] < self.pos[0] < self.max_pos[0] or not self.min_pos[1] < self.pos[1] < self.max_pos[1]:
            self.out_of_bound = True
        if self.timer > self.window_size - 1:
            h_states = self.all_pose_calculation(target, False)
            self.robot_pos = self.next_robot_pos(self.frame_duration, target, self.robot_pos)
            h_states1 = self.all_pose_calculation(target, False)
            pygame.draw.circle(self.screen, self.black, (int(self.robot_pos[0]), int(self.robot_pos[1])), 5, 0)
        intersect_ball = self.pos_prediction(self.robot_delta_t(target))
        dist = math.sqrt(((intersect_ball[0] - target[0]) * (intersect_ball[0] - target[0])) + (
                (intersect_ball[1] - target[1]) * (intersect_ball[1] - target[1])))
        done = False
        # if dist < 10:
        #     reward = 10
        #     done = True
        # else:
        reward = -0.01
        dist2 = math.sqrt(((self.pos[0] - self.robot_pos[0]) * (self.pos[0] - self.robot_pos[0])) + (
                (self.pos[1] - self.robot_pos[1]) * (self.pos[1] - self.robot_pos[1])))
        if dist2 < 10:
            reward = 10
            done = True
        if self.out_of_bound:
            done = True
        history = self.history.queue
        pygame.draw.circle(self.screen, self.green, (int(self.history.queue[0][0]), int(self.history.queue[0][1])), 5,
                           0)  # ballvis
        pygame.draw.circle(self.screen, self.blue, (int(self.history.queue[2][0]), int(self.history.queue[2][1])), 5,
                           0)  # ballvis
        if frame_num == 3:
            history[0][0] = history[0][0] / self.max_pos[0]
            history[0][1] = history[0][1] / self.max_pos[1]
            history[1][0] = history[1][0] / self.max_pos[0]
            history[1][1] = history[1][1] / self.max_pos[1]
        history[2][0] = history[2][0] / self.max_pos[0]
        history[2][1] = history[2][1] / self.max_pos[1]
        robot_pos = (self.robot_pos[0] / self.screen_size[0], self.robot_pos[1] / self.screen_size[1])
        dist = math.sqrt(((self.robot_pos[0] - target[0]) * (self.robot_pos[0] - target[0])) + (
                (self.robot_pos[1] - target[1]) * (self.robot_pos[1] - target[1])))
        normalized_dist = dist / 720
        return (history[0][0], history[0][1], history[1][0], history[1][1], history[2][0], history[2][1],
                normalized_dist), h_states, h_states1, reward, done

    def all_pose_calculation(self, target, next):
        t_robot = self.robot_delta_t(target)
        if next:
            t_robot = t_robot - self.frame_duration
        sizes = []
        for i in range(self.window_size):
            size_x = self.vel_init[0] * t_robot
            size_y = self.vel_init[1] * t_robot
            sizes.append(math.sqrt((size_x * size_x) + (size_y * size_y)))
            t_robot -= self.frame_duration
        pygame.draw.circle(self.screen, self.blue,
                           (int(target[0]), int(target[1])), 3, 0)
        positions = []
        dist = math.sqrt(((self.robot_pos[0] - target[0]) * (self.robot_pos[0] - target[0])) + (
                (self.robot_pos[1] - target[1]) * (self.robot_pos[1] - target[1])))
        normalized_dist = dist / 720
        for i in range(36):
            angle = (i * math.pi / 180) * 10
            temp_pos = []
            for i in range(self.window_size):
                temp_pos.append((sizes[i] * math.cos(angle) + target[0]) / self.max_pos[0])
                temp_pos.append((sizes[i] * math.sin(angle) + target[1]) / self.max_pos[1])
                if not next:
                    if i == 0:
                        pygame.draw.circle(self.screen, self.green,
                                           (int(sizes[i] * math.cos(angle) + target[0]),
                                            int(sizes[i] * math.sin(angle) + target[1])), 1, 0)
                    if i == 2:
                        pygame.draw.circle(self.screen, self.red,
                                           (int(sizes[i] * math.cos(angle) + target[0]),
                                            int(sizes[i] * math.sin(angle) + target[1])), 1, 0)
            robot_pos = (self.robot_pos[0] / self.screen_size[0], self.robot_pos[1] / self.screen_size[1])
            temp_pos.append(normalized_dist)
            positions.append(temp_pos)
        pygame.display.update()
        return np.array(positions)

    def test_situation(self):
        ball_poses = ((self.min_pos[0], self.min_pos[1]),  # 1
                      (self.min_pos[0] + ((self.max_pos[0] - self.min_pos[0]) / 2), self.min_pos[1]),  # 2
                      (self.max_pos[0], self.min_pos[1]),  # 3
                      (self.min_pos[0], self.min_pos[1] + ((self.max_pos[1] - self.min_pos[1]) / 2)),  # 4
                      (self.max_pos[0], self.min_pos[1] + ((self.max_pos[1] - self.min_pos[1]) / 2)),  # 5
                      (self.min_pos[0], self.max_pos[1]),  # 6
                      (self.min_pos[0] + ((self.max_pos[0] - self.min_pos[0]) / 2), self.max_pos[1]),  # 7
                      (self.max_pos[0], self.max_pos[1])  # 8
                      )
        angles_limits = ((0, 90), (0, 180), (90, 180), (-90, 90), (90, 270), (-90, 0), (180, 360), (180, 270))
        for i in range(len(ball_poses)):
            self.init_pos = position.Position2D(ball_poses[i][0], ball_poses[i][1])
            self.vel_init = (500, 1050)
            self.vector_ball = vector.Vector2D()
            self.pos = position.Position2D(ball_poses[i][0], ball_poses[i][1])
            timer = 0
            al = angles_limits[i]
            for cur_ang in range(al[0], al[1], 10):
                self.pos = self.init_pos
                timer = 0
                while self.min_pos[0] <= self.pos.x <= self.max_pos[0] and self.min_pos[1] <= self.pos.y <= \
                        self.max_pos[1]:
                    time.sleep(self.frame_duration)
                    timer += 1
                    self.vector_ball.from_angle_size(cur_ang * math.pi / 180,
                                                     self.vel_init[0] * timer * self.frame_duration)
                    self.pos = self.init_pos + self.vector_ball
                    # self.pos = self.pos_prediction(timer * self.frame_duration)
                    pygame.draw.circle(self.screen, self.red, (int(self.pos.x), int(self.pos.y)),
                                       self.window_size, 0)  # ballvis
                    pygame.display.update()

    def next_robot_pos(self, t, target, robot_pos):
        self.robot_speed = 200
        size = self.robot_speed * t
        motion_vector = np.array(target) - np.array(robot_pos)
        length = math.sqrt(motion_vector[0] * motion_vector[0] + motion_vector[1] * motion_vector[1])
        motion_vector = motion_vector / length
        motion_vector = motion_vector * size
        return robot_pos + motion_vector

    def main(self):
        self.init_velocity()
        self.init_position()
        # while True:
        # x = int(self.min_pos[0] + (random.random() * (self.max_pos[0] - self.min_pos[0])))
        # y = int(self.min_pos[1] + (random.random() * (self.max_pos[1] - self.min_pos[1])))
        # self.step((x, y))

    def test_queue(self):

        self.history.put(np.array((1, 1)))
        self.history.put(np.array((2, 1)))
        self.history.put(np.array((3, 1)))
        t = self.history.get()
        self.history.put(np.array((4, 1)))


if __name__ == "__main__":
    sim = simulator()
    sim.test_situation()
