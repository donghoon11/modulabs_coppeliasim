from dataclasses import dataclass
from abc import abstractmethod
import numpy as np


from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class LumiBot_sensor13:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

    def init_coppelia(self):
        self.lumiBot_ref = self.sim.getObject("/lumiBot_ref")
        self.lmotor = self.sim.getObjectHandle("./lumibot_leftMotor")
        self.rmotor = self.sim.getObjectHandle("./lumibot_rightMotor")
        
        self.sensorLF = self.sim.getObjectHandle('./LeftFront')
        self.sensorLR = self.sim.getObjectHandle('./LeftRear')
        self.sensorRF = self.sim.getObjectHandle('./RightFront')
        self.sensorRR = self.sim.getObjectHandle('./RightRear')
        self.sensorF = self.sim.getObjectHandle('./Forward')
        
        # params for driving
        self.min_d = 0.15
        self.max_d = 0.25
        self.yaw_cutoff = 0.005
        self.fwd_cutoff = 0.25
        
        self.avg_default = 0.15
        self.fwd_default = 100
        self.v = 0.5
        self.dv = 0.5
        self.v_sharp = 1
        self.v_straight = 2
        
        self.avg = self.avg_default
        self.diff = 0
        self.fwd = self.fwd_default

        # lidars
        self.lidars = []
        for i in range(1,14):
            self.lidars.append(self.sim.getObject(f"/lidar_{i:02d}"))
            # print(f"/lidar_{i:02d}")

    def read_lidars(self):
        scan = []
        for id in self.lidars:
            scan.append(self.sim.readProximitySensor(id))
        return scan
        
    def cleanup(self):
        pass

    def run_coppelia(self):
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        count = 0
        while self.run_flag:
            count += 1
            # step
            self.run_step(count)
            self.sim.step()
            # self.sysCall_actuation()
            # self.sysCall_sensing()
        # self.sim.stopSimulation()
    
    @abstractmethod
    def run_step(self, count):
        pass

if __name__ == "__main__" :
    client = LumiBot_sensor13()
    client.init_coppelia()
    client.run_coppelia()