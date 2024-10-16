from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

class MobileRobotPP:
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

        # KUKA youBot is a 4 mecanum wheel robot
        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),  # Front left
            self.sim.getObject('/rollingJoint_rl'),  # Rear left
            self.sim.getObject('/rollingJoint_rr'),  # Rear right
            self.sim.getObject('/rollingJoint_fr')   # Front right
        ]

        self.collVolumeHandle = self.sim.getObject('/youBot_coll')  # Collision volume
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

        self.robotObstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
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

    def set_movement(self, v_forward, v_turn, v_side):
        """
        Control the mecanum wheels, supporting lateral movement.
        :param v_forward: Forward/Backward velocity
        :param v_turn: Rotational velocity
        :param v_side: Lateral velocity (for side movement)
        """
        # 파라미터 설정
        radius = 0.05  # 휠 반지름
        distance_x = 0.228  # 로봇 중심에서 휠까지의 x 방향 거리
        distance_y = 0.158  # 로봇 중심에서 휠까지의 y 방향 거리
        dist_R = distance_x + distance_y


        theta_fr = 55   # deg
        theta_fl = 125  # deg
        theta_rl = 235  # deg
        theta_rr = 305  # deg


        # Calculate wheel velocities for mecanum drive
        fl_speed = (- v_forward - v_turn * dist_R - v_side) / radius
        rl_speed = (- v_forward - v_turn * dist_R + v_side) / radius
        rr_wheel_speed = (- v_forward + v_turn * dist_R - v_side) / radius
        fr_wheel_speed = (- v_forward + v_turn * dist_R + v_side) / radius

    
        # Set wheel velocities
        self.sim.setJointTargetVelocity(self.wheel_joints[0], fl_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], rl_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], rr_wheel_speed)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], fr_wheel_speed)

    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        symbol = True if collision == 0 else False
        return symbol

    def get_target_position(self):
        """Returns the position of the goal dummy object."""
        return self.sim.getObjectPosition(self.goalDummyHandle, -1)

    def visualize_path(self, path):
        """Visualizes the robot's path."""
        if self.line_container == None:  # Initial
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])

        # self.sim.addDrawingObject(self.line_container, None)

        if path:
            for i in range(1, len(path) // 2):
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

    def move_robot_to_position(self, target_position):
        path = None
        
        while not path:
            task = self.simOMPL.createTask('t')
            self.simOMPL.setAlgorithm(task, self.search_algo)

            start_pos = self.sim.getObjectPosition(self.refHandle, -1)

            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0] - self.search_range, start_pos[1] - self.search_range],
                                                [start_pos[0] + self.search_range, start_pos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
            self.simOMPL.setup(task)

            # 경로 단순화 (! 여러 path 중 가장 단순한 path = 최적화 경로)
            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)

            time.sleep(0.01)
        return path

    def follow_path(self, path):
        if path:
            path_3d = []
            for i in range(0, len(path) // 2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])

            prev_l = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                current_pos = self.sim.getObjectPosition(self.frontRefHandle, -1)
                path_lengths, total_dist = self.sim.getPathLengths(path_3d, 3)
                
                # 현재 위치와 가장 가까운 경로 찾기.
                closet_l = self.sim.getClosestPosOnPath(path_3d, path_lengths, current_pos)

                if closet_l <= prev_l:
                    closet_l += total_dist / 200
                prev_l = closet_l

                # 가장 가까운 위치에 대한 목표 지점 보간.
                target_point = self.sim.getPathInterpolatedConfig(path_3d, path_lengths, closet_l)
                self.sim.addDrawingObjectItem(track_pos_container, target_point)

                # Relative position of the target position
                m = self.sim.getObjectMatrix(self.refHandle, -1)
                self.sim.getMatrixInverse(m)

                # 현재 위치로부터 목표 지점의 상대 위치 계산
                relative_target = self.sim.multiplyVector(m, target_point)

                # Compute angle for rotation
                angle = math.atan2(relative_target[1], relative_target[0])

                # Forward/backward and
                #  rotation movement
                forward_velocity = 2.0
                turn_velocity = 4 * angle / math.pi

                # Set lateral velocity (for side movement) to 0 for now
                side_velocity = 0.0

                self.set_movement(forward_velocity, turn_velocity, side_velocity)

                # Stop when close to the target
                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.05:
                    break

                time.sleep(0.01)

    def run_step(self):
        self.sim.startSimulation()
        while self.run_flag:
            goal_position = self.get_target_position()

            # Adjust goal if collision is detected
            while self.check_collides_at(goal_position):
                goal_position[0] -= 0.09

            # Plan and follow the path
            print('1')
            path = self.move_robot_to_position(goal_position)
            if path:
                self.follow_path(path)

            # Stop movement
            self.set_movement(0, 0, 0)

            time.sleep(0.01)
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = MobileRobotPP()
    controller.init_coppelia()
    controller.run_step()
