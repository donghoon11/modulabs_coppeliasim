from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

@dataclass
class Control:
    vel_X: float = 0
    vel_Y: float = 0
    vel_Z: float = 0

class LumiBot:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True
        self.control = Control()

    def on_press(self, key):
        deltaX, deltaZ = 1.0, np.pi / 10
        if key == Key.up:
            self.control.vel_X += deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.down:
            self.control.vel_X -= deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.left:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z += deltaZ
        if key == Key.right:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z -= deltaZ
        self.control.vel_X = min(max(self.control.vel_X, -20), 20)
        self.control.vel_Y = 0
        self.control.vel_Z = min(max(self.control.vel_Z, -np.pi), np.pi)
        
    def init_coppelia(self):
        self.lumiBot_ref = self.sim.getObject("/lumiBot_ref")
        self.lmotor = self.sim.getObjectHandle("./lumibot_leftMotor")
        self.rmotor = self.sim.getObjectHandle("./lumibot_rightMotor")
        
        # self.sensorLF = self.sim.getObjectHandle('./LeftFront')
        # self.sensorLR = self.sim.getObjectHandle('./LeftRear')
        # self.sensorRF = self.sim.getObjectHandle('./RightFront')
        # self.sensorRR = self.sim.getObjectHandle('./RightRear')
        # self.sensorF = self.sim.getObjectHandle('./Forward')
        
        # # params for driving
        # self.min_d = 0.15
        # self.max_d = 0.25
        # self.yaw_cutoff = 0.005
        # self.fwd_cutoff = 0.25
        
        # self.avg_default = 0.15
        # self.fwd_default = 100
        # self.v = 0.5
        # self.dv = 0.5
        # self.v_sharp = 1
        # self.v_straight = 2
        
        # self.avg = self.avg_default
        # self.diff = 0
        # self.fwd = self.fwd_default

        # lidars
        self.lidars = []
        for i in range(1,6):
            self.lidars.append(self.sim.getObject(f"/lidar_{i:02d}"))
            # print(f"/lidar_{i:02d}")

    def control_car(self):
        self.sim.setJointTargetVelocity(
            self.lmotor,
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.rmotor,
            -self.control.vel_X - self.control.vel_Z,
        )

    def read_lidars(self):
        scan = []
        for id in self.lidars:
            scan.append(self.sim.readProximitySensor(id))
        return scan
    
    def run_coppelia(self):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        count = 0
        while self.run_flag:
            count += 1
            # step
            self.run_step(count)
            self.sim.step()
        self.sim.stopSimulation()
    
    @abstractmethod
    def run_step(self, count):
        pass

if __name__ == "__main__" :
    client = LumiBot()
    client.init_coppelia()
    client.run_coppelia()