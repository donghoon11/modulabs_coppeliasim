# zmq remote api 로 변환하기.
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class LumiBot:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

    def sysCall_init(self):
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
    
    # def get_object_handle(self, name):
    #     # Placeholder for the actual API call
    #     return sim.getObjectHandle(name)

    # def set_joint_target_velocity(self, handle, velocity):
    #     # Placeholder for the actual API call
    #     return self.sim.setJointTargetVelocity(handle, velocity)
    
    # def read_proximity_sensor(self, sensor):
    #     # Placeholder for the actual API call
    #     return self.sim.readProximitySensor(sensor)
    
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
        flag1, LF = self.sim.readProximitySensor(self.sensorLF)
        flag2, LR = self.sim.readProximitySensor(self.sensorLR)
        flag3, F = self.sim.readProximitySensor(self.sensorF)
        
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
        
        print(f'avg= {self.avg} diff= {self.diff} fwd= {self.fwd}')
    
    def cleanup(self):
        pass

    def run_coppelia(self):
        self.sim.startSimulation()
        while True:
            self.sysCall_actuation()
            self.sysCall_sensing()

if __name__ == "__main__" :
    client = LumiBot()
    client.sysCall_init()
    client.run_coppelia()