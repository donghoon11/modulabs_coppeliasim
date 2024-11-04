from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

class youBotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        # self.frontRefHandle = self.sim.getObject('/youBot_frontRef')
        self.collVolumeHandle = self.sim.getObject('/ME_Platfo2_sub1')
        self.goalDummyHandle = self.sim.getObject('/goalDummy')

        self.motor_fl = self.sim.getObject('/rollingJoint_fl')
        self.motor_rl = self.sim.getObject('/rollingJoint_rl')
        self.motor_fr = self.sim.getObject('/rollingJoint_fr')
        self.motor_rr = self.sim.getObject('/rollingJoint_rr')

        '''
        sim.createCollection()
        int options : bit 0 set (1): collection overrides collidable, measurable, detectable properties, and also the visibility state of its objects.
        Return : collectionHandle: handle of the new collection
        
        sim.addItemToCollection(int collectionHandle, int what, int objectHandle, int options)
        int options :
            bit 0 set (1): the specified object (or group of objects) is removed from the collection. Otherwise it is added.
            bit 1 set (2): the specified object is not included in the group of objects, if sim.handle_tree or sim.handle_chain is specified (i.e. the tree base or tip is excluded).
        '''
        self.robotObstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)     # "-1" means world?
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 180 * math.pi / 180
        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.5
        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

    def findPath(self, targetPos, path=None):
        while not path:
            task = self.simOMPL.createTask('t')
            self.simOMPL.setAlgorithm(task, self.search_algo)
            startPos = self.sim.getObjectPosition(self.refHandle, -1)
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [startPos[0] - self.search_range, startPos[1] - self.search_range],
                                                [startPos[0] + self.search_range, startPos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, startPos[:2])
            self.simOMPL.setGoalState(task, targetPos[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.01)
            self.simOMPL.setup(task)

            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
            time.sleep(0.01)
            print(path)
        return path

    def omni_wheel_control(self, v_forward, v_side, v_turn):
        # params for 4 mecanum wheel drive
        radius = 0.05       # wheel radius
        dist_R = 0.228 + 0.158      # (distance b.w. centroid & wheel cent.) = dist_x + dist_y

        # Calculate wheel velocities for mecanum drive
        # fl_speed = (-v_forward - v_side - v_turn * dist_R )
        # rl_speed = (-v_forward + v_side - v_turn * dist_R )
        # fr_speed = (-v_forward + v_side + v_turn * dist_R)
        # rr_speed = (-v_forward - v_side + v_turn * dist_R )
        fl_speed = (-v_forward - v_side - v_turn)
        rl_speed = (-v_forward + v_side - v_turn)
        fr_speed = (-v_forward + v_side + v_turn)
        rr_speed = (-v_forward - v_side + v_turn)

        # print(f"fl_spped : {fl_speed}")
        # print(f"rl_spped : {rl_speed}")
        # print(f"fr_spped : {fr_speed}")
        # print(f"rr_spped : {rr_speed}")
        # print()

        # Set motor velocities
        self.sim.setJointTargetVelocity(self.motor_fl, fl_speed)
        self.sim.setJointTargetVelocity(self.motor_rl, rl_speed)
        self.sim.setJointTargetVelocity(self.motor_fr, fr_speed)
        self.sim.setJointTargetVelocity(self.motor_rr, rr_speed)
    
    def followPath(self, path=None):
        if path:
            path_3d = []
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])
            
            prev_dist = 0
            while True:
                currPos = self.sim.getObjectPosition(self.refHandle, -1)

                pathLength, totalDist = self.sim.getPathLengths(path_3d, 3)
    
                closet_dist = self.sim.getClosestPosOnPath(path_3d, pathLength, currPos)

                if closet_dist <= prev_dist:
                    closet_dist += totalDist / 200
                prev_dist = closet_dist

                tartgetPoint = self.sim.getPathInterpolatedConfig(path_3d, pathLength, closet_dist)
                
                m = self.sim.getObjectMatrix(self.refHandle, -1)

                relative_target = self.sim.multiplyVector(m, tartgetPoint)
                print(f"relative_target : {relative_target}")
        
                angle = math.atan2(relative_target[1], relative_target[0])
                print(f'angle : {angle}')
                print()

                speedConfig = {'foward_vel' : 1.0, 
                               'turn_vel' : 4 * angle / math.pi,
                               'side_vel' : 0.0
                               }

                self.omni_wheel_control(speedConfig['foward_vel'], 
                                        speedConfig['side_vel'], speedConfig['turn_vel'])
                
                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.5:
                    break
                time.sleep(1)

    def run_coppelia(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation(True)
        while self.run_flag:
            goalPos = self.sim.getObjectPosition(self.goalDummyHandle, -1)
            print(f'goal position : {goalPos}')

            path = self.findPath(goalPos)
            self.omni_wheel_control(0.1,0.1,0.1)
            
            if path != None:
                self.followPath(path)
            time.sleep(1)
            self.omni_wheel_control(0.0, 0.0, 0.0)
            break
        print('robot reaches the goal position')
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = youBotPP()
    controller.init_coppelia()
    controller.run_coppelia()