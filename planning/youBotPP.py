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
        self.frontRefHandle = self.sim.getObject('/youBot_frontRef')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        self.goalDummyHandle = self.sim.getObject('/goalDummy')

        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),  # front left
            self.sim.getObject('/rollingJoint_rl'),  # rear left
            self.sim.getObject('/rollingJoint_fr'),   # front right
            self.sim.getObject('/rollingJoint_rr'),  # rear right
        ]

        self.prev_forwback_vel = 0
        self.prev_side_vel = 0
        self.prev_rot_vel = 0

        self.p_parm = 20
        self.max_v = 2
        self.p_parm_rot = 10
        self.max_v_rot = 3
        self.accel_f = 0.035    

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

        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1

        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

    def visualizePath(self, path):
        if self.line_container is None:
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])
        self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path)//2):
                line_data = [path[2*i-2], path[2*i -1], 0.001, path[2*i], path[2*i+1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

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
                self.simOMPL.simplifyPath(task, self.search_duration)       # search_duration 반경 넘어가면 simOMPL BiRRT 작동 에러.
                path = self.simOMPL.getPath(task)
                self.visualizePath(path)
            time.sleep(0.01)
        return path
    
    def followPath(self, path=None):
        if path:
            path_3d = []
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])
            print(f'path: {path_3d}')
            prev_dist = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                currPos = self.sim.getObjectPosition(self.frontRefHandle, -1)

                pathLength, totalDist = self.sim.getPathLengths(path_3d, 3)
                print(f'pathLength : {pathLength}')
                print(f'total distance : {totalDist}')
    
                closet_dist = self.sim.getClosestPosOnPath(path_3d, pathLength, currPos)
                print(f'closet_dist : {closet_dist}')

                if closet_dist <= prev_dist:
                    closet_dist += totalDist / 200
                prev_dist = closet_dist

                tartgetPoint = self.sim.getPathInterpolatedConfig(path_3d, pathLength, closet_dist)
                self.sim.addDrawingObjectItem(track_pos_container, tartgetPoint)
                print(f'targetPoint : {tartgetPoint}')
                
                m = self.sim.getObjectMatrix(self.frontRefHandle, -1)
                m_inv = self.sim.getMatrixInverse(m)

                rel_p = self.sim.multiplyVector(m_inv, tartgetPoint)
                rel_o = math.atan2(rel_p[1], rel_p[0]) - math.pi/2
                print(f'rel_p : {rel_p}')
                print(f'rel_o : {rel_o}')
                print()

                forwback_vel = rel_p[1] * self.p_parm
                side_vel = rel_p[0] * self.p_parm
                v = (forwback_vel**2 + side_vel**2)**0.5
                if v > self.max_v:
                    forwback_vel *= self.max_v / v
                    side_vel *= self.max_v / v

                rot_vel = -rel_o * self.p_parm_rot
                if abs(rot_vel) > self.max_v_rot :
                    rot_vel = self.max_v_rot * rot_vel / abs(rot_vel)

                df = forwback_vel - self.prev_forwback_vel
                ds = side_vel - self.prev_side_vel
                dr = rot_vel - self.prev_rot_vel

                if abs(df) > self.max_v * self.accel_f:
                    df = self.max_v * self.accel_f * df / abs(df)
                if abs(ds) > self.max_v * self.accel_f:
                    ds = self.max_v * self.accel_f * ds / abs(ds)
                if abs(dr) > self.max_v_rot * self.accel_f:
                    dr = self.max_v_rot * self.accel_f * dr / abs(dr)

                forwback_vel = self.prev_forwback_vel + df
                side_vel = self.prev_side_vel + ds
                rot_vel = self.prev_rot_vel + dr

                self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
                
                self.prev_forwback_vel = forwback_vel
                self.prev_side_vel = side_vel
                self.prev_rot_vel = rot_vel

                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.5:
                    
                    self.sim.removeDrawingObject(track_pos_container)
                    break
                self.sim.step()
                time.sleep(0.01)

    def omni_wheel_control(self, forwback_vel, side_vel, rot_vel):
        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
   

    def run_coppelia(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation(True)
        while self.run_flag:
            goalPos = self.sim.getObjectPosition(self.goalDummyHandle, -1)
            print(f'goal position : {goalPos}')

            path = self.findPath(goalPos)
            if path != None:
                self.followPath(path)
            time.sleep(0.1)
            self.omni_wheel_control(0.0, 0.0, 0.0)
            break
        self.sim.removeDrawingObject(self.line_container)
        print('robot reaches the goal position')
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = youBotPP()
    controller.init_coppelia()
    controller.run_coppelia()