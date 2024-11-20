from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

import matplotlib.pyplot as plt
from youBot import YouBot

import logging
import threading

class MappingBot(YouBot):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        # list eulerAngles = sim.getObjectOrientation(int objectHandle, 
        #                                     int relativeToObjectHandle = sim.handle_world)
        # eulerAngles: Euler angles [alpha beta gamma]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta
    
    def run_step(self):
        # self.control_car()
        
        scan = self.read_lidars()
        loc = self.read_ref()
        
        # update grid
        self.grid.update(loc, scan)

class Grid():
    def __init__(self):
        # grid 전체 배열 크기 100*100 으로 설정.
        self.grid = np.zeros((100, 100))
        # plot grid
        r = np.linspace(-5, 5, 101)  # -5부터 5까지 101개의 x 좌표 생성
        p = np.linspace(-5, 5, 101)  # -5부터 5까지 101개의 y 좌표 생성
        self.R, self.P = np.meshgrid(r, p)  # 2D 그리드 생성
        # plot object
        self.plt_objects = [None] * 15
        # scan theta, lidar setting 이므로 이 부분 수정하지 않아도 됨.
        self.delta = np.pi / 12     # 13개 라이다 -> 12개 간격으로 분해능.
        # -90도 기준으로 각 라이다별 스캔 각도 계산.
        self.scan_theta = np.array([- np.pi + self.delta * i for i in range(13)])
        self.boundary = np.pi / 2 + self.delta / 2      
        # min distance
        self.min_dist = (2 * (0.05**2)) ** 0.5  # 0.07071

    def update(self, loc, scan):
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    def mapping(self, loc, scan):
        x, y, theta = loc       # youBot_ref (x,y,yaw)
        theta += np.pi/2
        # scan position
        # 라이다 위치는 ref position (로봇 중앙) 으로부터 앞으로 0.275 m 으로 조정.
        rx = x + 0.275 * np.cos(theta)      # ref dummy 위치 몸체 중앙으로 변동되었으므로 0.275 m 값 수정 필요.
        ry = y + 0.275 * np.sin(theta)      # ref dummy 위치 몸체 중앙으로 변동되었으므로 0.275 m 값 수정 필요.
        # range
        dist = 2.25     # 스캔 센서 감지 최대 거리, scan data 테스트 통해서 distance 더 길게 할 수도 있음.
        '''
        i_min, i_max, j_min, j_max는 그리드 맵에서 스캔 범위 내의 인덱스를 결정. 
        그리드의 해상도는 0.1 미터이고, 그리드 맵이 100 x 100 크기를 가지며, 맵의 중앙이 (50,50)에 해당
        '''
        # 이 아래 부분은 sub grid search 를 하기 위해 조정한 부분.
        # xy_resolution = 0.1
        i_min = max(0, int((rx - dist) // 0.1 + 50))            # norm // resolution + map_center_index
        i_max = min(99, int((rx + dist) // 0.1 + 50))
        j_min = max(0, int((ry - dist) // 0.1 + 50))
        j_max = min(99, int((ry + dist) // 0.1 + 50))
        # sub grid
        sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]  # grid[row, col] = grid[y_range, x_range]

        # x distance
        # 1) i_min 부터 i_max 까지 정수 배열 생성 -> 0.1 해상도 곱하여 미터 단위 변환
        # 2) 배열 원소에 +0.05 함으로써 각 그리드 셀의 중심 좌표 구함.
        # 3) 그리드 크기가 10*10 이므로 0으로부터 상대적 중앙을 구하기 위해 -5
        gx = np.arange(i_min, i_max +1) * 0.1 + 0.05 - 5     # numpy.arange([start, ] stop, [step, ] dtype=None)
        gx = np.repeat(gx.reshape(1,-1), sub_grid.shape[0], axis=0)     # gx 배열을 1*N 형태로 변환.
        dx = gx - rx    # 라이다 위치로부터 sub grid 각 셀의 x 방향 거리 추정.
        # y distance
        gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
        gy = np.repeat(gy.reshape(1,-1).T, sub_grid.shape[1], axis=1)
        dy = gy - ry
        # distance
        gd = (dx**2 + dy**2) ** 0.5     # 라이다 위치로부터 sub grid 각 셀 까지의 거리 추정.

        # theta diff
        # 각 지점과 로봇의 스캔 세서 사이의 각도 차이를 구하고, 로봇이 바라보는 방향 theta 와 비교해 각도 차이 dtheta 를 계산.
        gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 -1)     # gtheta : global coord.origin 에 대한 광선의 각도
        dtheta = gtheta - theta                             # dtheta : 로봇의 yaw 를 고려한 광선의 상대 각도. 로봇 이동 방향 기준으로 계산된 광선의 각도.
        # 각도 범위 조정. (-pi ~ pi) 안에 존재할 수 있도록 설정. 단위 : [rad]
        while np.pi < np.max(dtheta):
            dtheta -= (np.pi < dtheta) * 2 * np.pi
        while np.min(dtheta) < -np.pi:
            dtheta += (dtheta < -np.pi) * 2 * np.pi
        
        # inverse sensor model : 역센서 모델
        '''
        Return values of 'sim.readProximitySensor'
        :   res = detection state (0 or 1)
            dist = distance to th detected point
            point = array of 3 numbers indicating the relative coord. of the detected point
            obj = handle of the object that was detected
            n = normal vector of the detected surface relative to the sensor reference frame. / finding norm vector of the detected surface rel. to the sensor ref.frame.
        '''
        # res 는 센서가 장애물을 감지했는지 여부, dist 는 장애물과의 거리
        for i in range(13):
            res, dist, _, _, _ = scan[i]        # 여기서 scan return 값에 대한 정의해주지 않으면 dist 에 튜플로 묶여서 값이 같이 들어가게됨.
            if res == 0:        # 장애물이 없는 경우 -> 해당 방향에서 특정 거리 내에 있는 셀들을 자유 공간으로 간주. free area
                area = (
                    (gd <= 2.25)    # sub gird 각 셀과 로봇 센서 사이의 거리가 2.25 미터 이하인지 확인.
                    * (-self.boundary + self.delta * i <= dtheta)       # i 번째 스캔 데이터가 커버하는 각도 범위 내에 있는 그리드 셀만을 업데이트 대상으로.
                    * (dtheta <= -self.boundary + self.delta * (i +1))
                )
                sub_grid[area] -= 0.5   # 위 조건을 만족하는 area 에서 그리드 셀의 값을 0.5 만큼 감소 -> 자유공간 가능성.
            else:       # 장애물 감지.
                dist = min(2.25, dist)      # dist 재정의.
                detect_area = (
                    (np.abs(gd - dist) < self.min_dist)     # 그리드 상의 각 지점과 로봇 센서 사이의 거리가 탐지된 거리와 매우 근접한지 여부 확인. 기준 값 : 0.07071
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[detect_area] += 0.5    # 그리드 값을 0.5만큼 증가 -> 장애물 있을 가능성 큼.
                free_area = (
                    (gd <= dist - self.min_dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[free_area] -= 0.5      # 장애물 있을 가능성 존재 -> 자유 공간 셀 가중치 감소.
        np.clip(self.grid, -5, 5, out=self.grid)        # 마지막으로 그리드의 모든 셀 값이 -5에서 5 사이로 제한.

    def save(self):
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/localization/mapping_test.npy", "wb") as f:
            np.save(f, self.grid)

    # visualizeing 부분은 matplotlib 만 다루므로 수정 하지 않아도 됨.
    def visualize(self, loc, scan):
        x, y, theta = loc
        theta += np.pi/2
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
        # robot
        (self.plt_objects[1],) = plt.plot(
            x, y, color="green", marker="o", markersize=10
        )
        # scan
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


class youBotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia_pp(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        # self.frontRefHandle = self.sim.getObject('/youBot_frontRef')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        #self.goalDummyHandle = self.sim.getObject('/goalDummy')
        #self.goalDummyHandle = self.sim.getObject('/balconyDummy')
        self.waypoints = [self.sim.getObject('/bedroom1'),
                          self.sim.getObject('/bedroom2'),
                          self.sim.getObject('/toilet'),
                          self.sim.getObject('/entrance'),
                          self.sim.getObject('/dining'),
                          self.sim.getObject('/livingroom'),
                          self.sim.getObject('/balcony_init'),
                          self.sim.getObject('/balcony_end'),
                        ]

        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),  # front left
            self.sim.getObject('/rollingJoint_rl'),  # rear left
            self.sim.getObject('/rollingJoint_fr'),   # front right
            self.sim.getObject('/rollingJoint_rr'),  # rear right
        ]

        self.prev_forwback_vel = 0
        self.prev_side_vel = 0
        self.prev_rot_vel = 0

        self.p_parm = 50 #20
        self.max_v = 10
        self.p_parm_rot = 10 #10
        self.max_v_rot = 3
        self.accel_f = 0.35

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

        self.search_range = 10
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

    def findPath(self, targetPos):
        path = None
        while not path:
            task = self.simOMPL.createTask('t')
            self.simOMPL.setAlgorithm(task, self.search_algo)
            startPos = self.sim.getObjectPosition(self.refHandle, -1)
            # statespace 탐색 결과에 대한 예외 처리 힐요.
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
    
    def followPath(self, goalDummyHandle, path=None):
        if path:
            path_3d = []
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])
            prev_dist = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                currPos = self.sim.getObjectPosition(self.refHandle, -1)

                pathLength, totalDist = self.sim.getPathLengths(path_3d, 3)
    
                closet_dist = self.sim.getClosestPosOnPath(path_3d, pathLength, currPos)

                if closet_dist <= prev_dist:
                    closet_dist += totalDist / 200
                prev_dist = closet_dist

                tartgetPoint = self.sim.getPathInterpolatedConfig(path_3d, pathLength, closet_dist)
                self.sim.addDrawingObjectItem(track_pos_container, tartgetPoint)
                
                m = self.sim.getObjectMatrix(self.refHandle, -1)
                m_inv = self.sim.getMatrixInverse(m)

                rel_p = self.sim.multiplyVector(m_inv, tartgetPoint)
                rel_o = math.atan2(rel_p[1], rel_p[0]) - math.pi/2      # yaw 조절하는 부분.

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

                if np.linalg.norm(np.array(self.sim.getObjectPosition(goalDummyHandle, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.6:
                    
                    self.sim.removeDrawingObject(track_pos_container)
                    break
                self.sim.step()
                # time.sleep(0.001)

    # omni_wheel_control 은 모터 속도 제어를 위해 추가한 메서드.
    def omni_wheel_control(self, forwback_vel, side_vel, rot_vel):
        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
   
    # re-naming : run_coppelia -> run_coppelia_pp
    def run_coppelia_pp(self):
        # self.sim.setStepping(True)
        self.init_coppelia_pp()     # 원래 클래스 외부의 main 함수에서 호출해야 하는데 thread 를 사용하므로 클래스 내부에서 호출.
        self.sim.startSimulation(True)
        while self.run_flag:
            for i in range(len(self.waypoints)):
                goalPos = self.sim.getObjectPosition(self.waypoints[i], -1)
                print(f'goal position : {goalPos}')
                try:
                    path = self.findPath(goalPos)
                    print(f'Find the path : waypoint{i}')
                except:
                    print('Fail to find path.')
                if path != None:
                    self.followPath(goalDummyHandle=self.waypoints[i], path=path)
                self.omni_wheel_control(0.0, 0.0, 0.0)
                self.sim.removeDrawingObject(self.line_container)
                self.line_container = None
                print('move to another waypoint')
                time.sleep(2)
            print('check')
            self.omni_wheel_control(0.0, 0.0, 0.0)
            #break
        print('robot reaches the goal position')
        self.sim.stopSimulation()

# thread 기능 설정.
if __name__ == "__main__":
    planning = youBotPP()
    mapping = MappingBot()

    thread_a = threading.Thread(target=planning.run_coppelia_pp)
    thread_b = threading.Thread(target=mapping.run_coppelia)

    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()