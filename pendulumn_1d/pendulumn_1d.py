import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Control():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        
        self.FSM_velocity = 0
        self.FSM_position = 1
        self.FSM_damping = 2


    def sysCall_init(self):

        # global joint
        # global FSM
        # global t_position
        # global t_damping

        
        self.joint = self.sim.getObject("/Joint")
        
        self.FSM = self.FSM_velocity
        self.t_position = 0
        self.t_damping = 0


    def sysCall_actuation(self):

        # 1) velocity control
        #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
        #sim.setJointTargetVelocity(joint, 0.5)
        
        # 2) position control
        #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        #sim.setJointTargetPosition(joint, math.pi/2)    # joint position using radian

        # 3) torque control
        #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
        #thetadot = sim.getJointVelocity(joint)
        #force = -0.5*thetadot       # dissipating energy
        #sim.setJointTargetForce(joint, force)
        
        t = self.sim.getSimulationTime()
        theta = self.sim.getJointPosition(self.joint)
        thetadot = self.sim.getJointVelocity(self.joint)
        
        if (self.FSM == self.FSM_velocity and theta > -0.5 and theta<0.5):
            self.FSM = self.FSM_position
            self.t_position = t
            print(f'position: {theta}')
            
        if (self.FSM == self.FSM_position and t-self.t_position > 3):
            self.FSM = self.FSM_damping
            self.t_damping = t
            print(f't : {t-self.t_position}')
            
        if (self.FSM == self.FSM_damping and thetadot > -0.1 and thetadot < 0.1 and t-self.t_damping > 4):
            self.FSM = self.FSM_velocity
            # we specify a minimum force at FSM_damping
            # and, that force persists in the memory
            # so we need to clear that force
            self.sim.setJointTargetForce(self.joint, 10000)
            print(f'thetadot : {thetadot}, SimulTime : {t}, T_interval : {t-self.t_damping} ')
        
    
        if (self.FSM == self.FSM_velocity):
            self.sim.setObjectInt32Param(self.joint, self.sim.jointintparam_dynctrlmode, self.sim.jointdynctrl_velocity)
            self.sim.setJointTargetVelocity(self.joint, 0.5)
        
        if (self.FSM == self.FSM_position):
            self.sim.setObjectInt32Param(self.joint, self.sim.jointintparam_dynctrlmode, self.sim.jointdynctrl_position)
            self.sim.setJointTargetPosition(self.joint, 0.1)    # joint position using radian    
            
        if (self.FSM == self.FSM_damping):
            self.sim.setObjectInt32Param(self.joint, self.sim.jointintparam_dynctrlmode, self.sim.jointdynctrl_force)
            
            force = -0.5*thetadot       # dissipating energy
            self.sim.setJointTargetForce(self.joint, force)       
        
    def run_coppelia(self):
        self.sim.startSimulation()
        while True:
            self.sysCall_actuation()
            
if __name__ == "__main__":
    client = Control()
    client.sysCall_init()
    client.run_coppelia()