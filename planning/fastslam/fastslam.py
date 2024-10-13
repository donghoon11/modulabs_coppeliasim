from env import VrepEnvironment     # ros 통신 -> joint control
import numpy as np
from sensor import Laser
import cv2
import time
from utils import relative2absolute, wrapAngle

import random
import copy
import os
import argparse
import yaml
import math

#from world import World
from robot import Robot
from world import World
from motion_model import MotionModel
from measurement_model import MeasurementModel
from utils import absolute2relative, relative2absolute, degree2radian, visualize, visualize_opencv
#import keyboard

class VrepEnvironment:
    def __init__(self) -> None:
        pass

### package sensor
class Laser(object):
    '''
    A class representing the specifications of a scanning laser rangefinder (Lidar).
    '''
    def __init__(self, scan_size, scan_rate_hz, detection_angle_degrees, distance_no_detection_mm, detection_margin=0, offset_mm=0):

        self.scan_size = scan_size
        self.scan_rate_hz = scan_rate_hz
        self.detection_angle_degrees = detection_angle_degrees
        self.distance_no_detection_mm = distance_no_detection_mm
        self.detection_margin = detection_margin
        self.offset_mm = offset_mm

    def __str__(self):

        return  'scan_size=%d | scan_rate=%3.3f hz | detection_angle=%3.3f deg | distance_no_detection=%7.4f mm | detection_margin=%d | offset=%4.4f m' % \
        (self.scan_size,  self.scan_rate_hz,  self.detection_angle_degrees, self.distance_no_detection_mm,  self.detection_margin, self.offset_mm)

    def __repr__(self):

        return str(self)


class URG04LX(Laser):
    '''
    A class for the Hokuyo URG-04LX
    '''
    def __init__(self, detectionMargin = 0, offsetMillimeters = 0):

        Laser.__init__(self, 682, 10, 240, 4000, detectionMargin, offsetMillimeters)

class XVLidar(Laser):
    '''
    A class for the GetSurreal XVLidar
    '''
    def __init__(self, detectionMargin = 0, offsetMillimeters = 0):

        Laser.__init__(self, 360, 5.5, 360, 6000, detectionMargin, offsetMillimeters)

class RPLidarA1(Laser):
    '''
    A class for the SLAMTEC RPLidar A1
    '''
    def __init__(self, detectionMargin = 0, offsetMillimeters = 0):

        Laser.__init__(self, 360, 5.5, 360, 12000, detectionMargin, offsetMillimeters)


## class utils:
class uitls:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))


    # Bresenhams Line Generation Algorithm
    # ref: https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
    def bresenham(x1, y1, x2, y2, w, h):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        steep = 0
        if dx <= dy:
            steep = 1
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            dx, dy = dy, dx

        pk = 2 * dy - dx

        loc = []
        for _ in range(0, dx + 1):
            if (x1 < 0 or y1 < 0) or (steep == 0 and (x1 >= h or y1 >= w)) or (steep == 1 and (x1 >= w or y1 >= h)):
                break

            if steep == 0:
                loc.append([x1, y1])
            else:
                loc.append([y1, x1])

            if x1 < x2:
                x1 = x1 + 1
            else:
                x1 = x1 - 1

            if (pk < 0):
                if steep == 0:
                    pk = pk + 2 * dy
                else:
                    pk = pk + 2 * dy
            else:
                if y1 < y2:
                    y1 = y1 + 1
                else:
                    y1 = y1 - 1

                pk = pk + 2 * dy - 2 * dx

        return loc


    def wrapAngle(radian):
        radian = radian - 2 * np.pi * np.floor((radian + np.pi) / (2 * np.pi))
        return radian


    def degree2radian(degree):
        return degree / 180 * np.pi


    def prob2logodds(prob):
        return np.log(prob / (1 - prob + 1e-15))


    def logodds2prob(logodds):
        return 1 - 1 / (1 + np.exp(logodds) + 1e-15)


    def normalDistribution(mean, variance):
        return np.exp(-(np.power(mean, 2) / variance / 2.0) / np.sqrt(2.0 * np.pi * variance))


    def create_rotation_matrix(theta):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        R_inv = np.linalg.inv(R)

        return R, R_inv


    def absolute2relative(position, states):
        x, y, theta = states
        pose = np.array([x, y])

        R, R_inv = create_rotation_matrix(theta)
        position = position - pose
        position = np.array(position) @ R_inv.T

        return position


    def relative2absolute(position, states):
        x, y, theta = states
        pose = np.array([x, y])

        R, R_inv = create_rotation_matrix(theta)
        position = np.array(position) @ R.T
        position = position + pose

        return position
    # def compute_odometry(v_left, v_right)

    def visualize(robot, particles, best_particle, radar_list, step, title, output_path, visualize=False):
        ax1.clear()
        ax2.clear()
        fig.suptitle("{}\n\n number of particles:{}, step:{}".format(title, len(particles), step + 1))
        ax1.set_title("Estimated by Particles")
        ax2.set_title("Ground Truth")
        ax1.axis("off")
        ax2.axis("off")

        grid_size = best_particle.grid_size
        ax1.set_xlim(0, grid_size[1])
        ax1.set_ylim(0, grid_size[0])

        grid_size = robot.grid_size
        ax2.set_xlim(0, grid_size[1])
        ax2.set_ylim(0, grid_size[0])

        # draw map
        world_map = 1 - best_particle.grid
        ax1.imshow(world_map, cmap='gray')
        world_map = 1 - robot.grid
        ax2.imshow(world_map, cmap='gray')

        # draw radar beams
        for (x, y) in radar_list:
            ax2.plot(x, y, "yo", markersize=1)

        # draw tragectory
        true_path = np.array(robot.trajectory)
        ax2.plot(true_path[:, 0], true_path[:, 1], "b")
        estimated_path = np.array(best_particle.trajectory)
        ax1.plot(estimated_path[:, 0], estimated_path[:, 1], "g")

        # draw particles position
        for p in particles:
            ax1.plot(p.x, p.y, "go", markersize=1)

        # draw robot position
        ax2.plot(robot.x, robot.y, "bo")

        if step % 10 == 0:
            plt.savefig('{}_{}.png'.format(output_path, step), bbox_inches='tight')

        if visualize:
            plt.draw()
            plt.pause(0.01)

    def visualize_opencv(robot, particles, best_particle, radar_list, step, title, output_path, recorder):
        world_map = 1 - best_particle.grid
        empty_map = np.ones((150, 150))
        img = np.stack((world_map,)*3, axis=-1)
        img1 = np.stack((empty_map,)*3, axis=-1)

        # draw particle
        for p in particles:
            cv2.circle(img, (int(p.x), int(p.y)), 1, (0, 0, 128), 1)
        # draw robot position
        cv2.circle(img1, (int(robot.x), int(robot.y)), 3, (0, 0, 255), 1)
        # draw robot orientation
        x = robot.x + np.cos(robot.theta)*3
        y = robot.y + np.sin(robot.theta)*3
        cv2.line(img1, (int(robot.x), int(robot.y)),(int(x), int(y)), (0, 0, 255), 1)
        # draw center map
        cv2.circle(img, (75, 75), 1, (255, 0, 0), 1)
        cv2.circle(img1, (75, 75), 1, (255, 0, 0), 1)
        # draw 1 m2 square grid
        for i in (50, 100):
            cv2.line(img, (0, i), (150, i), (0, 255, 0), 1)
            cv2.line(img, (i, 0), (i, 150), (0, 255, 0), 1)
            cv2.line(img1, (0, i), (150, i), (0, 255, 0), 1)
            cv2.line(img1, (i, 0), (i, 150), (0, 255, 0), 1)

        for (x, y) in radar_list:
            cv2.circle(img1, (x, y), 1, (128, 128, 0), 1)

        img = cv2.resize(img, (300,300))
        img1 = cv2.resize(img1, (300, 300))

        concated_img = np.concatenate((img, img1), axis=1)
        print(concated_img.shape)
        cv2.imshow("slam", concated_img)
        recorder.write(cv2.cvtColor((concated_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))


        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break








if __name__ == "__main__":

    env = VrepEnvironment(speed=1, turn=0.5, rate=100)
    RotationMatrix = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    scale_factor = 10
    floor_w = floor_h = 15*scale_factor

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ROBOT = config['robot']
    SCENE = config['scene-1']
    NUMBER_OF_PARTICLES = 100

    # create an unknow map
    init_grid = np.ones(SCENE['grid_size']) * ROBOT['prior_prob']

    # init robot
    (x, y, theta) = SCENE['R_init']
    R = Robot(x, y, theta, init_grid, ROBOT, sense_noise=3.0)
    prev_odo = curr_odo = R.get_state()

    p = [None] * NUMBER_OF_PARTICLES
    (x, y, theta) = SCENE['p_init']

    for i in range(NUMBER_OF_PARTICLES):
        p[i] = Robot(x, y, degree2radian(theta), copy.deepcopy(init_grid), ROBOT)

    # create motion model
    motion_model = MotionModel(config['motion_model'])

    # create measurement model
    measurement_model = MeasurementModel(config['measurement_model'], ROBOT['radar_range'])
    output_path = "result/"

    idx = 0
    # create video recorder
    w = 600
    h = 300
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recorder = cv2.VideoWriter("result/map.mp4", fourcc, fps, (w, h))
    #while True:
    for i in range(100):
        #time.sleep(1)
        action = np.random.choice(2)
        #event = keyboard.read_event()
        #R.action2move(action, env.v_forward, env.v_turn, env.rate)
        curr_odo = R.get_state()
        R.update_trajectory()

        # if input("Please enter a string:\n") == "w":
        #     print("moving forward")
        #     action = 1
        # elif input("Please enter a string:\n") == "a":
        #     print("turning left")
        #     action = 0
        #
        # elif input("Please enter a string:\n") == "d":
        #     print("turning right")
        #     action = 2

        print("take action", action)
        transform, lidar_data = env.step(action=action)

        pos = transform.translation
        qua = transform.rotation
        robot_pos = np.array((pos.x, pos.y, pos.z))*scale_factor
        #robot_pos = RotationMatrix @ robot_pos + 75
        robot_pos = robot_pos + 75
        robot_pos_xy = (int(robot_pos[0]), int(robot_pos[1]))
        # robot_theta_w = 2 *np.arcsin(qua.z)
        # robot_theta = -2 *np.arcsin(qua.z) - np.pi/2
        robot_theta_w =qua.z
        #robot_theta = -2 *np.arcsin(qua.z) - np.pi/2
        R.x, R.y, R.theta = (int(robot_pos[0]), int(robot_pos[1]), robot_theta_w)

        # print("robot_pos", robot_pos_xy)
        # print("theta", robot_theta)
        # print("result from velocity")
        print(R.x, R.y, R.theta)
        robot_state = (pos.x, pos.y, robot_theta_w)

        scan = np.reshape(lidar_data, (270, -1))
        z_star, free_grid_star, occupy_grid_star = R.sense(lidar_data=scan, robot_state=robot_state)

        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)
        w = np.zeros(NUMBER_OF_PARTICLES)
        for i in range(NUMBER_OF_PARTICLES):
            prev_pose = p[i].get_state()
            x, y, theta = motion_model.sample_motion_model(prev_odo, curr_odo, prev_pose)
            p[i].set_states(x, y, theta)
            p[i].update_trajectory()

            # Calculate particle's weights depending on robot's measurement
            z, _, _ = p[i].sense()
            w[i] = measurement_model.measurement_model(z_star, z)

            # Update occupancy grid based on the true measurements
            curr_pose = p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            p[i].update_occupancy_grid(free_grid, occupy_grid)

        # normalize
        w = w / np.sum(w)
        best_id = np.argsort(w)[-1]

        # select best particle
        estimated_R = copy.deepcopy(p[best_id])

        # Resample the particles with a sample probability proportional to the importance weight
        # Use low variance sampling method
        new_p = [None] * NUMBER_OF_PARTICLES
        J_inv = 1 / NUMBER_OF_PARTICLES
        r = random.random() * J_inv
        c = w[0]

        i = 0
        for j in range(NUMBER_OF_PARTICLES):
            U = r + j * J_inv
            while (U > c):
                i += 1
                c += w[i]
            new_p[j] = copy.deepcopy(p[i])

        p = new_p
        prev_odo = curr_odo
        print("vis")
        #estimated_R = copy.deepcopy(R)
        # print(p)
        # print(R)
        # print(free_grid_star)
        visualize_opencv(R, p, estimated_R, free_grid_star, idx, "FastSLAM 1.0", output_path, recorder)
        idx += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
