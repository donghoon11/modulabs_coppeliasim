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

        # KUKA youBot 은 4 mecanum wheel 기반으로 differential drive 기반의 로봇과는 차이가 있다.
        self.leftMotorHandle = self.sim.getObject('/rollingJoint_fl')
        self.rightMotorHandle = self.sim.getObject('/rollingJoint_fr')

        self.collVolumeHandle = self.sim.getObject('/youBot_coll')     # dr12 의 boundary box
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

        self.robotObstaclesCollection = self.sim.createCollection(0)        # 충돌 검사용 객체 컬렉션
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 180 * math.pi / 180     # 바퀴 회전 속도
        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1
        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None


    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        return collision


    def get_target_position(self):
        """Returns the position of the goal dummy object."""
        return self.sim.getObjectPosition(self.goalDummyHandle, -1)
    
    
    def visualize_path(self, path):
        """Visualizes the robot's path."""
        if not self.line_container:     # 초기
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])

        self.sim.addDrawingObject(self.line_container, None)

        if path:
            for i in range(1, len(path)/2):
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)


    def move_robot_to_position(self, target_position):
        path = None
        while not path:
            task = self.simOMPL.createTask('t') 
            self.simOMPL.setAlgorithm(task, self.search_algo)

            start_pos = self.sim.getObjectPosition(self.refHandle, -1)
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0]-self.search_range, start_pos[1]-self.search_range],
                                                [start_pos[0]+self.search_range, start_pos[1]+self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
            self.simOMPL.setup(task)

            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)

            time.sleep(0.01)
        return path
    
    
    def follow_path(self, path):
        if path:
            path_3d = []
            for i in range(0, len(path)/2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])

            prev_l = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                current_pos = self.sim.getObjectPosition(self.frontRefHandle, -1)
                path_lengths, total_dist = self.sim.getPathLenghts(path_3d, 3)
                closet_l = self.sim.getClosetPosOnPath(path_3d, path_lengths, current_pos)
                
                if closet_l <= prev_l:
                    closet_l += total_dist / 200
                prev_l = closet_l

                target_point = self.sim.getPathInterpolatedConfig(path_3d, path_lengths, closet_l)
                self.sim.addDrawingObjectItem(track_pos_container, target_point)

                # Relative position of the target position
                m = self.sim.getObjectMatrix(self.refHandle, -1)
                self.sim.inverseMatrix(m)
                relative_target = self.sim.multiplyVector(m, target_point)

                # Compute angle
                angle = math.atan2(relative_target[1], relative_target[0])

                # Adjust wheel velocities
                left_v = 1.0 - 4 * angle / math.pi if angle > 0 else 1.0
                right_v = 1.0 + 4 * angle / math.pi if angle <= 0 else 1.0
                left_v = max(-1.0, left_v)
                right_v = max(-1.0, right_v)

                self.sim.setJointTargetVelocity(self.leftMotorHandle, left_v * self.velocity)
                self.sim.setJointTargetVelocity(self.rightMotorHandle, right_v * self.velocity)

                # Break when close to the target
                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) - 
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.05:
                    break

                time.sleep(0.01)


    def run_step(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            goal_position = self.get_target_position()

            # Adjust goal if collision is detected:
            while self.check_collides_at(goal_position):
                goal_position[0] -= 0.09

            # Plan and follow the path
            path = self.move_robot_to_position(goal_position)
            if path:
                self.follow_path(path)

            # Stop
            self.sim.setJointTargetVelocity(self.leftMotorHandle, 0.0)
            self.sim.setJointTargetVelocity(self.rightMotorHandle, 0.0)

            time.sleep(0.01)
        self.sim.stopSimulation()


if __name__ == "__main__":
    controller = MobileRobotPP()
    controller.init_coppelia()
    controller.run_step()
