import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Pendulumn():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim') 

        self.T_end = 10
        self.r= 0.5

    def curve(self, t):
        f = 2*math.pi/self.T_end 
        
        x_ref = self.x_center + self.r*math.cos(f*t)
        y_ref = self.y_center + self.r*math.sin(f*t)
        
        # x_ref = x_center + r --> x_center = x -r
        # y_ref = y_center --> y_center = y
        
        return x_ref, y_ref
        

    def sysCall_init(self):

        # global joint1
        # global joint2
        # global endeff
        # global endeff_trace
        # global x_center
        # global y_center
        
        self.joint1 = self.sim.getObject("/Joint1")
        self.joint2 = self.sim.getObject("/Joint2")
        
        self.endeff = self.sim.getObject("/EndEff")
        self.endeff_trace = self.sim.addDrawingObject(self.sim.drawing_linestrip,5,0,-1,100000,[1,0,0])
        
        self.endeff_position = self.sim.getObjectPosition(self.endeff, -1)

        self.x = self.endeff_position[0]
        self.y = self.endeff_position[1]
        self.x_center = self.x - self.r
        self.y_center = self.y

        
    def sysCall_actuation(self):

        theta1 = self.sim.getJointPosition(self.joint1)   # radian
        theta2 = self.sim.getJointPosition(self.joint2)   # radian
        print("theta1 : "+str(theta1)+", theta2 : "+str(theta2))
        
        # dq = Jinv*dr = JInv*(X_ref - X)
        l = 1
        J = np.array([[l*(np.cos(theta1) + np.cos(theta1 + theta2)), l*np.cos(theta1 + theta2)], \
                        [l*(np.sin(theta1) + np.sin(theta1 + theta2)), l*np.sin(theta1 + theta2)]])
        
        Jinv = np.linalg.inv(J)
        
        t = self.sim.getSimulationTime()
        x_ref, y_ref = self.curve(t)
        #X_ref = [x_ref, y_ref]
        endeff_position = self.sim.getObjectPosition(self.endeff, -1)
        x = endeff_position[0]
        y = endeff_position[1]
        #X = [x, y]
        dr = [x_ref - x, y_ref - y]
        dq = Jinv.dot(dr)
        
        theta1 += dq[0]
        theta2 += dq[1] 
        self.sim.setJointTargetPosition(self.joint1, theta1)
        self.sim.setJointTargetPosition(self.joint2, theta2)


    def sysCall_sensing(self):
        # is executed before a simulation starts
        
        endeff_position = self.sim.getObjectPosition(self.endeff, self.sim.handle_world)
        self.sim.addDrawingObjectItem(self.endeff_trace, endeff_position)

    def run_coppelia(self):
        # start simulation
        self.sim.startSimulation()
        while self.sim.getSimulationTime() < self.T_end:
            self.sysCall_actuation()
            self.sysCall_sensing()
        self.sim.stopSimulation()

if __name__ == "__main__" :
    client = Pendulumn()
    client.sysCall_init()
    client.run_coppelia()