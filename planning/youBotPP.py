from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time
# from youBot_controll import MecanumWheelController


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
        # self.leftMotorHandle = self.sim.getObject('/rollingJoint_fl')
        # self.rightMotorHandle = self.sim.getObject('/rollingJoint_fr')
        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),     # wheel_joints[0]
            self.sim.getObject('/rollingJoint_rl'),     # wheel_joints[1]
            self.sim.getObject('/rollingJoint_rr'),     # wheel_joints[2]
            self.sim.getObject('/rollingJoint_fr')      # wheel_joints[3]
        ]

        self.collVolumeHandle = self.sim.getObject('/youBot_coll')     # dr12 의 boundary box
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

        self.robotObstaclesCollection = self.sim.createCollection(0)        # 충돌 검사용 객체 컬렉션
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


    # 충돌 검사 메서드
    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)

        # 충돌 쌍을 사용하여 충돌 여부를 확인 -> return bool
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        return collision

    # 목표 더미 객체의 위치를 반환한다.
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
                # 현재 포인트와 이전 포인트를 연결하는 라인 데이터를 생성.
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)


    def move_robot_to_position(self, target_position):
        path = None     # 경로 초기화
        while not path:
            task = self.simOMPL.createTask('t')     # 새로운 경로 생성.
            self.simOMPL.setAlgorithm(task, self.search_algo)

            start_pos = self.sim.getObjectPosition(self.refHandle, -1)  # 로봇의 현재 위치
            
            # 상태 공간 생성 : 2d 공간에서 로봇이 탐색할 상태 공간을 생성한다.
            # 상태 공간의 경계를 설정한다ㅣ.
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0]-self.search_range, start_pos[1]-self.search_range],
                                                [start_pos[0]+self.search_range, start_pos[1]+self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            
            # 유효성 검사 해상도 설정.
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
            self.simOMPL.setup(task)

            if self.simOMPL.solve(task, self.search_duration):
                # 경로 단순화
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)

            time.sleep(0.01)    # 경로 생성 과정에서 잠시 대기.
        return path
    
    
    def follow_path(self, path):
        if path:
            path_3d = []    # 경로를 3d 좌표로 변환할 리스트 초기화 (*이 부분도 np.로 최적화할 수 있지 않을까 생각.)
            # 경로를 2d 포인트에서 3d 포인트로 변환하기 위해 반복.
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])       # 각 2d 포인트에 z좌표 0을 추가하여 3d 좌표로 변환.

            prev_l = 0  # 이전 경로 위치 초기화
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                current_pos = self.sim.getObjectPosition(self.frontRefHandle, -1)
                path_lengths, total_dist = self.sim.getPathLenghts(path_3d, 3)
                
                # 현재 위치와 가장 가까운 경로의 위치를 찾는다.
                closet_l = self.sim.getClosetPosOnPath(path_3d, path_lengths, current_pos)
                
                # 이전 위치보다 가까운 경우를 체크.
                if closet_l <= prev_l:
                    # 경로를 부드럽게 하기 위해 최소 거리 조정.
                    closet_l += total_dist / 200
                prev_l = closet_l   # 위치 업데이트

                # 가장 가까운 위치에 대한 목표 지점을 보간한다.
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
                # 속도가 -1보다 낮아지지 않도록 제한.
                left_v = max(-1.0, left_v)
                right_v = max(-1.0, right_v)

                self.sim.setJointTargetVelocity(self.leftMotorHandle, left_v * self.velocity)
                self.sim.setJointTargetVelocity(self.rightMotorHandle, right_v * self.velocity)

                # Break when close to the target when the distance becomes 0.05
                if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) - 
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.05:
                    break

                time.sleep(0.01)        # 매 루프 사이에 잠시 대기.


    def run_step(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation()
        count = 0
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
