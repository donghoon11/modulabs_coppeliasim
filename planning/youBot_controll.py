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

    def set_movement(self, forw_back_vel, left_right_vel, rot_vel):
        """
        Control the mecanum wheels with forward/backward, left/right, and rotation velocities.
        
        :param forw_back_vel: Forward/Backward velocity
        :param left_right_vel: Left/Right strafing velocity
        :param rot_vel: Rotation velocity
        """
        # Apply the desired wheel velocities to the mecanum wheels
        # 로봇 팔이 있는 방향을 앞이라고 한다면, 앞을 향하려면 모든 바퀴가 CW 로 회전해야함.
        # 따라서 joint 방향에 (-) 를 해줘야함.
        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forw_back_vel - left_right_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forw_back_vel + left_right_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forw_back_vel - left_right_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forw_back_vel + left_right_vel + rot_vel)

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
            self.move_forward(1.0, 3)     # Move forward at speed 1.0 for 3 seconds
            # self.strafe_left(0.5, 2)      # Strafe left at speed 0.5 for 2 seconds
            # self.rotate(1.0, 2)           # Rotate at speed 1.0 for 2 seconds
            # self.move_backward(1.0, 3)    # Move backward at speed 1.0 for 3 seconds
            # self.strafe_right(0.5, 2)     # Strafe right at speed 0.5 for 2 seconds
        self.sim.stopSimulation()


if __name__ == '__main__':
    # Instantiate the MecanumWheelController class
    robot = MecanumWheelController()
    robot.init_coppelia()
    robot.run_coppelia()
