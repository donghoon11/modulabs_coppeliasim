from dataclasses import dataclass
from abc import abstractmethod
import numpy as np


from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class LumiBot:
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
        for i in range(1,6):
            self.lidars.append(self.sim.getObject(f"/lidar_{i:02d}"))
            # print(f"/lidar_{i:02d}")

    def read_lidars(self):
        scan = []
        for id in self.lidars:
            scan.append(self.sim.readProximitySensor(id))
        return scan

    def sysCall_actuation(self):
        self.sim.setJointTargetVelocity(self.lmotor, self.v_straight)
        self.sim.setJointTargetVelocity(self.rmotor, self.v_straight)
        
        if self.fwd < self.fwd_cutoff:
            print('going toward the wall, turn right')
            self.sim.setJointTargetVelocity(self.lmotor, self.v_sharp)
            self.sim.setJointTargetVelocity(self.rmotor, 0)
        elif self.fwd > self.fwd_cutoff:
            if self.avg > self.max_d:
                print('going away from the wall, turn left')
                self.sim.setJointTargetVelocity(self.lmotor, self.v - self.dv)
                self.sim.setJointTargetVelocity(self.rmotor, self.v)
            elif self.avg < self.min_d:
                print('going toward the wall, turn right')
                self.sim.setJointTargetVelocity(self.lmotor, self.v)
                self.sim.setJointTargetVelocity(self.rmotor, self.v - self.dv)
            elif self.min_d < self.avg < self.max_d:
                if self.diff > self.yaw_cutoff:  # LF > LR
                    print('yaw correction: turn left')
                    self.sim.setJointTargetVelocity(self.lmotor, self.v - self.dv)
                    self.sim.setJointTargetVelocity(self.rmotor, self.v)
                elif self.diff < -self.yaw_cutoff:  # LF < LR
                    self.sim.setJointTargetVelocity(self.lmotor, self.v)
                    self.sim.setJointTargetVelocity(self.rmotor, self.v - self.dv)
    
    def sysCall_sensing(self):
        # flag1, LF = self.sim.readProximitySensor(self.sensorLF)
        # flag2, LR = self.sim.readProximitySensor(self.sensorLR)
        # flag3, F = self.sim.readProximitySensor(self.sensorF)
        flag1, LF, _, _, _ = self.sim.readProximitySensor(self.sensorLF)
        flag2, LR, _, _, _ = self.sim.readProximitySensor(self.sensorLR)
        flag3, F, _, _, _ = self.sim.readProximitySensor(self.sensorF)
        
        if flag1 == 0 and flag2 == 1:
            self.avg = LR
            self.diff = 0
        elif flag1 == 1 and flag2 == 0:
            self.avg = LF
            self.diff = 0
        elif flag1 == 1 and flag2 == 1:
            self.avg = 0.5 * (LF + LR)
            self.diff = LF - LR
        else:
            self.avg = self.avg_default
            self.diff = 0
        
        if flag3 == 1:
            self.fwd = F
        else:
            self.fwd = self.fwd_default
        
        # print(f'avg= {self.avg} diff= {self.diff} fwd= {self.fwd}')
    
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
            self.sysCall_actuation()
            self.sysCall_sensing()
    
    @abstractmethod
    def run_step(self, count):
        pass

if __name__ == "__main__" :
    client = LumiBot()
    client.init_coppelia()
    client.run_coppelia()