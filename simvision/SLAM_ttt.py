import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class RobotController:
    def __init__(self):
        # Initialize CoppeliaSim remote API
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.wheel_radius = 0.195 / 2
        self.b = 0.1655
        self.vref = 0.35
        self.e = 0.24
        self.k = 1

        self.not_first_here = False
        
    def init_coppelia(self):
        # Get object handles
        # Pioneer P3DX : robot_pose, fastHokuyo
        self.left_wheel = self.sim.getObject('Pioneer_p3dx_leftMotor')
        self.right_wheel = self.sim.getObject('Pioneer_p3dx_rightMotor')
        self.robot_pose = self.sim.getObject('robot_pose')
        self.hokuyo = self.sim.getObject('fastHokuyo')
        # path : ref_point
        self.path = self.sim.getObject('Path')
        self.ref_point = self.sim.getObject('ref_point')
        
        # Initialize robot pose and grid map
        self.pose = self.update_robot_pose()
        # setPathTargetNominalVelocity : api func. 
        self.sim.setPathTargetNominalVelocity(self.path, self.vref)
        
        self.simGridMap = self.sim.callScriptFunction('init@simGridMap', self.sim.scripttype_customizationscript,
                                                      {0.1, 50, 0.9, 0.1, 10, 0.1, 0.1, 0.8, 0.7, 0.3}, True)

    def cleanup(self):
        # Release resources
        self.sim.callScriptFunction('release@simGridMap', self.sim.scripttype_customizationscript)

    # 로봇의 현재 위치와 속도를 기반으로 주행 궤적 추정.
    # kinemaitc_control() 을 통해 각 휠의 목표 속도 설정.
    def actuation(self):
        # Perform kinematic control
        ptraj, vtraj = self.get_trajectory_point()
        poff, voff = self.get_off_center_point(ptraj, vtraj, self.e)
        wL, wR = self.kinematic_control(poff, voff, self.pose, self.k)
        
        # Set the joint target velocities
        self.sim.setJointTargetVelocity(self.left_wheel, wL)
        self.sim.setJointTargetVelocity(self.right_wheel, wR)

    def sensing(self):
        # Get laser points and update robot pose
        laser_points = self.get_laser_points()
        self.pose = self.update_robot_pose()
        # self.sim.callScriptFunction('updateMapLaser@simGridMap', self.sim.scripttype_customizationscript,
        #                             [laser_points, self.pose])

    # hokuyo 센서에서 레이저 스캔 데이터를 가져와 리스트로 변환.
    # sensing() 메서드에서 업데이트.
    def get_laser_points(self):
        # Get laser scan data from Hokuyo sensor
        laser_scan = self.sim.callScriptFunction('getMeasuredData@fastHokuyo', self.sim.scripttype_childscript)
        laser_pts = []
        print(laser_scan)
        for i in range(0, len(laser_scan), 3):
            laser_pts.append([laser_scan[i], laser_scan[i + 1]])
        
        return laser_pts

    def get_trajectory_point(self):
        # Get reference point's position, orientation, and velocity
        position = self.sim.getObjectPosition(self.ref_point, -1)
        orientation = self.sim.getObjectOrientation(self.ref_point, -1)
        linear_vel, angular_vel = self.sim.getObjectVelocity(self.ref_point)

        # Adjust orientation and prepare trajectory data
        if orientation[2] > 0:
            ptraj = [position[0], position[1], orientation[2] - math.pi / 2]
        else:
            ptraj = [position[0], position[1], math.pi / 2 - orientation[2]]

        vtraj = [linear_vel[0], linear_vel[1], angular_vel[2]]
        
        return ptraj, vtraj

    def get_off_center_point(self, ptraj, vtraj, e):
        # Compute off-center point and velocity
        xc = ptraj[0] + e * math.cos(ptraj[2])
        yc = ptraj[1] + e * math.sin(ptraj[2])
        vxc = vtraj[0] - e * vtraj[2] * math.sin(ptraj[2])
        vyc = vtraj[1] + e * vtraj[2] * math.cos(ptraj[2])
        
        return [xc, yc], [vxc, vyc]

    def kinematic_control(self, ptraj, vtraj, pose, k):
        # Perform kinematic control based on trajectory and robot pose
        ex = ptraj[0] - (pose[0] + self.e * math.cos(pose[2]))
        ey = ptraj[1] - (pose[1] + self.e * math.sin(pose[2]))
        vxc = vtraj[0] + k * ex
        vyc = vtraj[1] + k * ey
        
        wL = (1 / (self.e * self.wheel_radius)) * ((self.e * math.cos(pose[2]) + self.b * math.sin(pose[2])) * vxc +
                                                   (self.e * math.sin(pose[2]) - self.b * math.cos(pose[2])) * vyc)
        wR = (1 / (self.e * self.wheel_radius)) * ((self.e * math.cos(pose[2]) - self.b * math.sin(pose[2])) * vxc +
                                                   (self.e * math.sin(pose[2]) + self.b * math.cos(pose[2])) * vyc)
        return wL, wR

    def update_robot_pose(self):
        # Update the robot's pose (position and orientation)
        position = self.sim.getObjectPosition(self.robot_pose, -1)
        orientation = self.sim.getObjectOrientation(self.robot_pose, -1)
        return [position[0], position[1], orientation[2]]

    def run_coppelia(self):
        # Start simulation and loop through actuation and sensing
        self.sim.setStepping(True)
        self.sim.startSimulation()
        
        while True:
            self.sensing()
            self.actuation()
            self.sim.step()
        
        self.sim.stopSimulation()


if __name__ == "__main__":
    robot_controller = RobotController()
    robot_controller.run_coppelia()
