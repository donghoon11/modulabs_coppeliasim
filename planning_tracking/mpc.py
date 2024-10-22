from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

class PP:
    def __init__(self) -> None:
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
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

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
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)     # -1 : 마지막 object = cuboid(벽)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 180 * math.pi / 180
        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1
        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)

        # 특정 pos 로 이동한 다음 충돌 검사.
        self.sim.setObjectPosition(self.collVolumeHandle, pos, -1)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        print(collision)        # collision[0] 결과가 0이면 충돌.

        # 다시 원래위치로 옮겨오기.
        self.sim.setObjectPosition(self.collVolumeHandle, tmp, -1)
        symbol = True if collision[0] == 0 else False
        return symbol
    
    def get_target_position(self):
        return self.sim.getObjectPosition(self.goalDummyHandle, -1)
    
    def visualize_path(self, path:list):
        '''
        sim.addDrawingObject(int objectType, float size, float duplicateTolerance, 
                            int parentObjectHandle, int maxItemCount, list color = None)
        '''
        if self.line_container is None:
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2,0.2,0.2])
        # self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path) // 2):
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

    def move_robot_to_position(self, target_position, path=None):
        
        while not path:
            task = self.simOMPL.createTask('t')
            
            # path planning 수행하는 과정
            self.simOMPL.setAlgorithm(task, self.search_algo)
            start_pos = self.sim.getObjectPosition(self.robotHandle, -1)
            
            '''
            string stateSpaceHandle = simOMPL.createStateSpace(string name, int type, int objectHandle, 
                                                                list boundsLow, list boundsHigh, int useForProjection, 
                                                                float weight=1.0, int refObjectHandle=-1)
            '''
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0] - self.search_range, start_pos[1] - self.search_range],
                                                [start_pos[0] + self.search_range, start_pos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
            self.simOMPL.setup(task)

            # 경로 단순화
            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                print(f'path : {path}')
                self.visualize_path(path)

            time.sleep(0.01)
        return path
    
    def omni_wheel_control(self, v_forward, v_side, v_turn):
        # params for 4 mecanum wheel drive
        radius = 0.05       # wheel radius
        dist_R = 0.228 + 0.158      # (distance b.w. centroid & wheel cent.) = dist_x + dist_y

        # Calculate wheel velocities for mecanum drive
        fl_speed = (-v_forward - v_side - v_turn * dist_R ) / radius
        rl_speed = (-v_forward + v_side - v_turn * dist_R ) / radius
        fr_speed = (-v_forward + v_side + v_turn * dist_R) / radius
        rr_speed = (-v_forward - v_side + v_turn * dist_R ) / radius

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

    def follow_path(self, path):
        if path:
            path_3d = []
            for i in range(0, len(path)//2) :
                path_3d.extend([path[2*i], path[2*i+1], 0.0])       # 3차원 좌표로 수정

            prev_l = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                current_pos = self.sim.getObjectPosition(self.frontRefHandle, -1)
                path_length, total_dist = self.sim.getPathLengths(path_3d, 3)   # list path, int dof

                closet_l = self.sim.getClosestPosOnPath(path_3d, path_length, current_pos)

                if closet_l <= prev_l:
                    closet_l += total_dist / 200
                prev_l = closet_l

                target_point = self.sim.getPathInterpolatedConfig(path_3d, path_length, closet_l)
                self.sim.addDrawingObjectItem(track_pos_container, target_point)

                m = self.sim.getObjectMatrix(self.refHandle, -1)
                self.sim.getMatrixInverse(m)
                
                relative_target = self.sim.multiplyVector(m, target_point)

                angle = math.atan2(relative_target[1], relative_target[0])

                forward_vel = 8.0
                turn_vel = 4 * angle / math.pi
                side_vel = 0.0

                self.omni_wheel_control(forward_vel, side_vel, turn_vel)

                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.05:
                    break
                
                # time.sleep(0.01)

    def run_step(self):
        self.sim.startSimulation()
        while self.run_flag:
            goal_position = self.get_target_position()
            while self.check_collides_at(goal_position):
                # 충돌 보정
                goal_position[0] -= 0.01

            path = self.move_robot_to_position(goal_position)
            if path:
                self.follow_path(path)
            
            time.sleep(0.01)
            print('2')
            self.omni_wheel_control(0.0,0.0,0.0)
            break
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = PP()
    controller.init_coppelia()
    controller.run_step()
