import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class PendulumControl():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim') 

    def sysCall_init(self):

        self.L1 = 1.0
        self.L2 = 1.0
        
        self.joint1 = self.sim.getObject("/Joint1")
        self.joint2 = self.sim.getObject("/Joint2")
        
        self.endeff = self.sim.getObject("/EndEff")
        self.endeff_trace = self.sim.addDrawingObject(self.sim.drawing_linestrip,5,0,-1,100000,[1,0,0])
            
        self.endeff_position = self.sim.getObjectPosition(self.endeff, -1)

        self.x = self.endeff_position[0]
        self.y = self.endeff_position[1]

        self.start_point = np.array([1,1])
        self.end_point = np.array([0,-1.5])

        # velocity and time interval
        self.velocity = 0.5      # m/sec
        self.time_interval = 0.01    # sec

    # IK
    def IK(self,xc,yc):
        # Using Law of cosine ; a^2 = b^2+c^2 - 2*b*c*cos(theta)
        cos_theta2 = (xc**2 + yc**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2)
        sin_theta2 = np.sqrt(1-cos_theta2**2)
        theta2 = np.arctan2(sin_theta2, cos_theta2)

        k1 = self.L1 + self.L2*cos_theta2
        k2 = self.L2 * sin_theta2
        
        theta1 = np.arctan2(yc,xc) - np.arctan2(k2,k1)

        return theta1, theta2
    
    # Jacobian
    def jacobian_matrix(self, theta1, theta2):
        J11 = - self.L1 * np.sin(theta1) - self.L2 * np.sin(theta1+theta2)
        J12 = - self.L2 * np.sin(theta1 + theta2)
        J21 = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        J22 = self.L2 * np.cos(theta1 + theta2)
        J = np.array([[J11, J12],[J21, J22]])
        return J

    def sysCall_actuation(self):

        # theta1 = self.sim.getJointPosition(self.joint1)   # radian
        # theta2 = self.sim.getJointPosition(self.joint2)   # radian
        distance = np.linalg.norm(self.end_point - self.start_point)

        total_time = distance / self.velocity
        time_vector = np.arange(0, total_time, self.time_interval)

        # target position init
        positions = np.zeros((len(time_vector), 2))

        direction_vector = (self.end_point - self.start_point) / distance

        # Calculate target position
        for i, t in enumerate(time_vector):
            current_position = self.start_point + self.velocity*t*direction_vector
            positions[i, :] = current_position


        # joints angles init
        joint_angles = np.zeros((len(time_vector), 2))

        # Calculate joints angles at target positions
        for i in range(len(time_vector)):
            x = positions[i, 0]
            y = positions[i, 1]
            theta1, theta2 = self.IK(x,y)
            joint_angles[i, :] = [theta1, theta2]

        # velocity of end effector
        endeff_vel = self.velocity + direction_vector

        # velocity of joints init
        joint_vels = np.zeros((len(time_vector),2))

        for i in range(len(time_vector)):
            theta1 = joint_angles[i, 0]
            theta2 = joint_angles[i, 1]
            J = self.jacobian_matrix(theta1, theta2)
            joint_vels[i, :] = np.linalg.inv(J) @ endeff_vel

        # Theta1 속도 플롯
        plt.figure()
        plt.plot(time_vector, joint_vels[:, 0])
        plt.xlabel('Time (s)')
        plt.ylabel('Theta1 Velocity (rad/s)')
        plt.title('Joint Velocity Theta1')
        plt.grid(True)
        plt.show()

        # Theta2 속도 플롯
        plt.figure()
        plt.plot(time_vector, joint_vels[:, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Theta2 Velocity (rad/s)')
        plt.title('Joint Velocity Theta2')
        plt.grid(True)
        plt.show()

    def sysCall_sensing(self):
        # is executed before a simulation starts
        self.sim.addDrawingObjectItem(self.endeff_trace, self.endeff_position)


    def run_coppelia(self):
        # start simulation
        self.sim.startSimulation()
        while True:
            # if self.endeff_positio == self.end_point:
            #     break
            self.sysCall_actuation()
            self.sysCall_sensing()
        self.sim.stopSimulation()

if __name__ == "__main__" :
    client = PendulumControl()
    client.sysCall_init()
    client.run_coppelia()