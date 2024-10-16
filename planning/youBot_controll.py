import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class MecanumWheelController:
    def __init__(self):
        # Create a connection to CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

    def init_coppelia(self):
        # Retrieve handles for the wheel joints (front left, rear left, rear right, front right)
        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),     # wheel_joints[0]
            self.sim.getObject('/rollingJoint_rl'),     # wheel_joints[1]
            self.sim.getObject('/rollingJoint_rr'),     # wheel_joints[2]
            self.sim.getObject('/rollingJoint_fr')      # wheel_joints[3]
        ]
        self.youBot_ref = self.sim.getObject('/youBot_ref')


    def set_movement(self, v_forward, v_side, v_turn):
        """
        Control the mecanum wheels with forward/backward, left/right, and rotation velocities.
        
        :param forw_back_vel: Forward/Backward velocity
        :param left_right_vel: Left/Right strafing velocity
        :param rot_vel: Rotation velocity
        """
        # Apply the desired wheel velocities to the mecanum wheels
        # 로봇 팔이 있는 방향을 앞이라고 한다면, 앞을 향하려면 모든 바퀴가 CW 로 회전해야함.
        # 따라서 joint 방향에 (-) 를 해줘야함.

        # 파라미터 설정
        radius = 0.05  # 휠 반지름
        distance_x = 0.228  # 로봇 중심에서 휠까지의 x 방향 거리
        distance_y = 0.158  # 로봇 중심에서 휠까지의 y 방향 거리
        dist_R = distance_x + distance_y

        alpha_fr = 55 * math.pi / 180   # deg
        alpha_fl = 125 * math.pi / 180  # 
        alpha_rl = 235 * math.pi / 180  # deg
        alpha_rr = 305 * math.pi / 180  # deg

        # x_dot = v_side
        # y_dot = -v_forward
        # theta_dot = v_turn
        # x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]

        # eulerAngles: Euler angles [alpha beta gamma]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        theta = theta * math.pi / 180
        print(f'theta : {theta}')
        # 각도 예외처리 해줘야함.
        sin_theta_fr = np.sin(theta + alpha_fr)
        sin_theta_fl = np.sin(theta + alpha_fl)
        sin_theta_rl = np.sin(theta + alpha_rl)
        sin_theta_rr = np.sin(theta + alpha_rr)

        cos_theta = np.cos(theta)
        cos_theta_fr = np.cos(theta + alpha_fr) * cos_theta
        cos_theta_fl = np.cos(theta + alpha_fl) * cos_theta
        cos_theta_rl = np.cos(theta + alpha_rl) * cos_theta
        cos_theta_rr = np.cos(theta + alpha_rr) * cos_theta


        fl_speed = (- v_forward * cos_theta_fl - v_turn * dist_R - v_side * sin_theta_fl) / radius
        rl_speed = (- v_forward * cos_theta_rl - v_turn * dist_R + v_side * sin_theta_rl) / radius
        rr_speed = (- v_forward * cos_theta_rr + v_turn * dist_R - v_side * sin_theta_rr) / radius
        fr_speed = (- v_forward * cos_theta_fr + v_turn * dist_R + v_side * sin_theta_fr) / radius

        self.sim.setJointTargetVelocity(self.wheel_joints[0], fl_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], rl_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], rr_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], fr_speed)

    def move_forward(self, speed, duration):
        """
        Move the robot forward at the specified speed for a certain duration.
        
        :param speed: Speed of the forward movement
        :param duration: Duration of the movement (in seconds)
        """
        self.set_movement(speed, 0, 0)
        time.sleep(duration)
        self.stop()

    def move_backward(self, speed, duration):
        """
        Move the robot backward at the specified speed for a certain duration.
        
        :param speed: Speed of the backward movement
        :param duration: Duration of the movement (in seconds)
        """
        self.set_movement(-speed, 0, 0)
        time.sleep(duration)
        self.stop()

    def strafe_left(self, speed, duration):
        """
        Strafe the robot to the left at the specified speed for a certain duration.
        
        :param speed: Speed of the strafing movement
        :param duration: Duration of the movement (in seconds)
        """
        self.set_movement(0, speed, 0)
        time.sleep(duration)
        self.stop()

    def strafe_right(self, speed, duration):
        """
        Strafe the robot to the right at the specified speed for a certain duration.
        
        :param speed: Speed of the strafing movement
        :param duration: Duration of the movement (in seconds)
        """
        self.set_movement(0, -speed, 0)
        time.sleep(duration)
        self.stop()

    def rotate(self, speed, duration):
        """
        Rotate the robot at the specified speed for a certain duration.
        
        :param speed: Speed of the rotation movement
        :param duration: Duration of the movement (in seconds)
        """
        self.set_movement(0, 0, speed)
        time.sleep(duration)
        self.stop()

    def stop(self):
        """
        Stop all wheel movements.
        """
        self.set_movement(0, 0, 0)


    def run_coppelia(self):
        self.sim.startSimulation()
        while self.run_flag:
            # Example movements
            self.move_forward(0.5, 2)     # Move forward at speed 1.0 for 3 seconds
            self.strafe_left(0.5, 1)      # Strafe left at speed 0.5 for 2 seconds
            self.rotate(1.0, 1)           # Rotate at speed 1.0 for 2 seconds
            self.move_backward(1.0, 1)    # Move backward at speed 1.0 for 3 seconds
            self.strafe_right(0.5, 1)     # Strafe right at speed 0.5 for 2 seconds
        self.sim.stopSimulation()


if __name__ == '__main__':
    # Instantiate the MecanumWheelController class
    robot = MecanumWheelController()
    robot.init_coppelia()
    robot.run_coppelia()
