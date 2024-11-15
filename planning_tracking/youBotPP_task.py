from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

import logging
import threading

import collections
import cv2
from IPython.display import clear_output

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pynput import keyboard
from pynput.keyboard import Key, Listener

from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

# path planning & tracking & wheel vel. ctrl
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
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        self.goalDummyHandles = [self.sim.getObject('/goalDummy'),
                                self.sim.getObject('/bedroom')]
        #self.goalDummyHandle = self.sim.getObject('/balconyDummy')
        # self.waypoints = [self.sim.getObject('/room1Dummy'),
        #                   self.sim.getObject('/room2Dummy'),
        #                   self.sim.getObject('/entranceDummy'),
        #                 ]

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
            task = self.simOMPL.createTask(f't')
            self.simOMPL.setAlgorithm(task, self.search_algo)
            startPos = self.sim.getObjectPosition(self.refHandle, -1)
            # rstatespace 탐색 결과에 대한 예외 처리 힐요.
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [startPos[0] - self.search_range, startPos[1] - self.search_range],
                                                [startPos[0] + self.search_range, startPos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, startPos[:2])
            self.simOMPL.setGoalState(task, targetPos[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.01)
            self.simOMPL.setup(task)
            print('10')
            if self.simOMPL.solve(task, self.search_duration):
                print('11')
                self.simOMPL.simplifyPath(task, self.search_duration)       # search_duration 반경 넘어가면 simOMPL BiRRT 작동 에러.
                path = self.simOMPL.getPath(task)
                self.visualizePath(path)
            else: 
                # path
                self.omni_wheel_control(0.2,0.2,0.2)
            time.sleep(0.01)
        return path
    
    def followPath(self, goalDummy, path=None):
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

                if np.linalg.norm(np.array(self.sim.getObjectPosition(goalDummy, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.6:
                    
                    self.sim.removeDrawingObject(track_pos_container)
                    break
                self.sim.step()
                # time.sleep(0.001)

    def omni_wheel_control(self, forwback_vel, side_vel, rot_vel):
        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
   
    def clear_path(self):
        self.sim.removeDrawingObject(self.line_container)
        self.line_container = None
'''
    def run_coppelia(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation(True)
        while self.run_flag:
            for i in range(2):
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
                    self.clear_path()
                    if i == 1:
                        break
                    self.redBoxDummy = self.sim.getObject('/goalDummy')
                    redBoxPos = self.sim.getObjectPosition(self.redBoxDummy, -1)
                    print(f'target position : {redBoxPos}')

                    try:
                        path_l = self.findPath(redBoxPos)
                        print(f'Find the path')
                    except:
                        print('Fail to find path.')
                    if path_l != None:    
                        self.followPath(goalDummyHandle=self.redBoxDummy, path=path_l)
                        self.omni_wheel_control(0.0, 0.0, 0.0)
                        self.clear_path()
                print('move to another waypoint')
                time.sleep(2)
            print('check')
            #self.omni_wheel_control(0.0, 0.0, 0.0)
            #break
        print('robot reaches the goal position')
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = youBotPP()
    controller.init_coppelia()
    controller.run_coppelia()
'''
# pich & place
PI_HALF = np.pi / 2

class TargetDetector:
    def __init__(self, client):
        self.client = client
        self.visual_objects = [None] * 20
        self.stage = 0      # 실행해야하는 과정을 나타내는 flag
        self.count = 0

    def detect_red_box(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        # 마스크를 이용해 원본 이미지에서 빨간색 부분 추출
        # 컨투어 찾기
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 원본 이미지에 컨투어 그리기
        bboxs = []
        for contour in contours:
            # 경계 상자 그리기
            x, y, w, h = cv2.boundingRect(contour)
            bboxs.append((x, y, w, h))
        return bboxs

    def find_nearest_bbox(self, base_x, zero_y, bboxs):
        near_zero_d = 0xFFFFFFFF
        near_index = -1
        for  i, (x, y, w, h) in enumerate(bboxs):
            center_x = x + w // 2
            center_y = y + h // 2
            zero_d = (base_x - center_x)**2 + (zero_y - center_y)**2
            if zero_d < near_zero_d:
                near_zero_d = zero_d
                near_index = i
        return near_index

    def control_find_target(self):
        if self.stage == 0 or self.stage == 2 or self.stage == 4:
            if self.client.thetas[0] >= np.pi / 2:
                self.stage += 1     # 90도 이상 회전 시 stage 한단계 증가.
            else:
                self.client.thetas[0] += 0.01
        elif self.stage == 1 or self.stage == 3 or self.stage == 3:
            if self.client.thetas[0] <= -np.pi / 2:
                self.stage += 1
                self.client.thetas[1] += np.pi / 16
            else:
                self.client.thetas[0] -= 0.01
        elif self.stage == 1 or self.stage == 3 or self.stage == 5:
            if self.client.thetas[0] <= -np.pi / 2:
                self.stage += 1
                self.client.thetas[1] += np.pi / 16
            else:
                self.client.thetas[0] -= 0.01

    # 카메라에 중심에 target이 오도록 조절
    def control_base_target(self, base_x, base_y, bbox):
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        diff_x = center_x - base_x
        diff_y = center_y - base_y
        if abs(diff_x) > 4 or abs(diff_y) > 4:
            self.client.thetas[0] -= np.clip(diff_x / 100, -0.01, 0.01)
            self.client.thetas[1] -= np.clip(diff_y / 100, -0.01, 0.01)
            return False
        return True
    
    def calc_target(self):
        _, pc_hat = self.client.fk(self.client.thetas[1:],
                                  [self.client.thetas[0]])
        xyz = R.from_quat(pc_hat[3:]).as_euler('xyz')
        theta_z = np.pi + xyz[0]
        theta_x = xyz[2]
        dist = pc_hat[2] * np.tan(theta_z)
        pt_hat = pc_hat[:3] + dist * np.array([-np.sin(theta_x), np.cos(theta_x), 0])
        pt_hat[-1] = 0.02
        return pt_hat

    # 시각화
    def visualize(self, img, bboxs, near_index):
        # 초기화
        for i in range(len(self.visual_objects)):
            if self.visual_objects[i] is None:
                break
            self.visual_objects[i].remove()
            self.visual_objects[i] = None
        # 이미지 출력
        self.visual_objects[0] = plt.imshow(img)
        for  i, (x, y, w, h) in enumerate(bboxs):
            rect = patches.Rectangle((x, y), w, h,
                                     linewidth=1,
                                     edgecolor='g' if i == near_index else 'k',
                                     facecolor='none')
            self.visual_objects[i + 1] = plt.gca().add_patch(rect)
        plt.pause(0.001)

    # 카메라를 이용해 target 검출
    def find_target(self, visualize=False):
        img = self.client.read_camera_1()
        base_x = img.shape[1] // 2
        base_y = img.shape[0] // 2
        zero_y = img.shape[0]

        bboxs = self.detect_red_box(img)

        if len(bboxs) == 0:
            self.control_find_target()

        near_index = self.find_nearest_bbox(base_x, zero_y, bboxs)
        flag = False
        if len(bboxs) > 0 and near_index >= 0:
            flag = self.control_base_target(base_x, base_y, bboxs[near_index])
            self.stage = 0
    
        if visualize:
            self.visualize(img, bboxs, near_index)
        
        pt_hat = None
        if flag:
            pt_hat = self.calc_target()
        return pt_hat

class Kinematics:
    def __init__(self, client):
        self.client = client
        self.stage = 0
        self.target_thetas = None
        self.place_thetas = [np.pi, -np.pi / 6, -np.pi / 2.7, -np.pi / 3, 0]
        self.base_thetas = [0.0, -np.pi / 6, -np.pi / 3, -np.pi / 3, 0.0]

    def ik(self, thetas, params):
        pt = params[-1][:3]
        pe_hat, _ = self.client.fk(thetas, params)
        # theta 범위 검증
        if thetas[0] < np.deg2rad(-90) or np.deg2rad(75) < thetas[0]:
            return 10, 0, 0, 0
        elif thetas[1] < np.deg2rad(-131.00) or np.deg2rad(131.00) < thetas[1]:
            return 10, 0, 0, 0
        elif thetas[2] < np.deg2rad(-102.00) or np.deg2rad(102.00) < thetas[2]:
            return 10, 0, 0, 0
        elif thetas[3] < np.deg2rad(-90.00) or np.deg2rad(90.00) < thetas[3]:
            return 10, 0, 0, 0
        return np.linalg.norm(pe_hat - pt), 0, 0, 0
    
    def solve(self, js, pt):
        target_thetas = fsolve(
            self.ik,
            [js[1], js[2], js[3], js[4]],
            [js[0], pt]
        )
        target_thetas[3] = 0
        return np.concatenate((np.array([js[0]]), target_thetas))

    def calc_target(self, pt_hat):
        target_thetas = self.solve(self.client.thetas, pt_hat)
        return target_thetas
    
    def trace_joint(self, target_thetas):
        diff_sum = 0
        for i, target in enumerate(target_thetas):
            diff = self.client.thetas[i] - target
            diff_sum += abs(diff)
            self.client.thetas[i] -= np.clip(diff, -0.05, 0.05)
        return diff_sum

    def pick_and_place(self, pt_hat):
        if self.stage == 0:
            self.target_thetas = self.calc_target(pt_hat)
            self.stage += 1
        elif self.stage == 1:
            diff_sum = self.trace_joint(self.target_thetas)
            if diff_sum < 0.01: self.stage += 1
        elif self.stage == 2:
            self.stage += 1
        elif self.stage < 13:
            self.client.control_gripper(True)
            self.stage += 1
        elif self.stage == 13:
            diff_sum = self.trace_joint([0, 0, -np.pi / 3, -np.pi / 3, 0])
            if diff_sum < 0.01: self.stage += 1
        elif self.stage == 14:
            diff_sum = self.trace_joint([np.pi, 0, -np.pi / 3, -np.pi / 3, 0])
            if diff_sum < 0.01: self.stage += 1
        elif self.stage == 15:
            diff_sum = self.trace_joint(self.place_thetas)
            if diff_sum < 0.01: self.stage += 1
        elif self.stage < 26:
            self.client.control_gripper(False)
            self.stage += 1
        elif self.stage == 26:
            diff_sum = self.trace_joint([np.pi, 0, -np.pi / 3, -np.pi / 3, 0])
            if diff_sum < 0.01: self.stage += 1
        elif self.stage == 27:
            diff_sum = self.trace_joint(self.base_thetas)
            if diff_sum < 0.01:
                self.stage = 0
                self.target_thetas = None
                return True
        return False
    
class PickAndPlace(youBotPP):
    def __init__(self):
        super().__init__()
        # 클라이언트 생성
        client = RemoteAPIClient()
        # 시뮬레이션 객체 가져오기
        self.sim = client.require("sim")
        # 바퀴
        # self.wheels = []
        # wheel angles
        self.angles = [0.0, 0.0, 0.0, 0.0]
        # 관절 (joint-0 ~ joint-4, grupper1, gripper2)
        self.joints = []
        # joint angles
        self.thetas = [0.0, -np.pi / 6, -np.pi / 3, -np.pi / 3, 0.0]
        # camera 1
        self.camera_1 = None
        # target information (image, bbox)
        self.target = None
        # run flag
        self.run_flag = True
        # Helper Class
        self.targetDetector = TargetDetector(self)
        self.kinematics = Kinematics(self)

        # # Multi threading Class
        # self.pathplanner = youBotPP()

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char("q"):
            self.run_flag = False

    def init_coppelia(self):
        # Wheel Joints are already defined in class "youBotPP"

        # Set params for path planning & tracking
        self.init_coppelia_pp()

        # 5개의 관절 가져오기
        for i in range(5):
            self.joints.append(self.sim.getObject(f"/youBotArmJoint{i}"))

        # Gripper Joint
        self.joints.append(self.sim.getObject(f"/youBotGripperJoint1"))
        self.joints.append(self.sim.getObject(f"/youBotGripperJoint2"))

        # camera
        self.camera_1 = self.sim.getObject(f"/camera_1")

        # joint 제어 모드 변경
        for joint in self.joints:
            self.sim.setObjectInt32Param(
                joint,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_position,
            )

    # control_wheel 에서 호출
    def init_wheelmode_posctrl(self):
        # wheel 제어 모드 변경
        for wheel in self.wheel_joints:
            self.sim.setObjectInt32Param(
                wheel,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_position,     # vel. 제어 -> pos 제어
            )
    # control_wheel 에서 호출
    def init_wheelmode_velctrl(self):
        # wheel 제어 모드 변경
        for wheel in self.wheel_joints:
            self.sim.setObjectInt32Param(
                wheel,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_velocity,     # pos. 제어 -> vel. 제어
            )
    # joint angle 조회
    def read_joints(self):
        js = []
        for joint in self.joints:
            j = self.sim.getJointPosition(joint)
            js.append(j)
        return js
    
    # wheel position 제어
    def control_wheel_pos(self):
        self.init_wheelmode_posctrl()       # 휠 속도 제어에서 위치 제어로 변경.
        for wheel, j in zip(self.wheel_joints, self.angles):
            self.sim.setJointTargetPosition(wheel, j)
    
    # joint 제어
    def control_joint(self):
        for joint, j in zip(self.joints, self.thetas):
            self.sim.setJointTargetPosition(joint, j)
    
    # gripper 제어
    def control_gripper(self, state):
        p1 = self.sim.getJointPosition(self.joints[-2])
        p2 = self.sim.getJointPosition(self.joints[-1])
        p1 += -0.005 if state else 0.005
        p2 += 0.005 if state else -0.005
        self.sim.setJointTargetPosition(self.joints[-2], p1)
        self.sim.setJointTargetPosition(self.joints[-1], p2)
        return p2
    
    # camera image 조회
    def read_camera_1(self):
        result = self.sim.getVisionSensorImg(self.camera_1)
        img = np.frombuffer(result[0], dtype=np.uint8)
        img = img.reshape((result[1][1], result[1][0], 3))
        img = cv2.flip(img, 1)
        return img
    
    # forward kinematics
    def fk(self, thetas, params):
        j1, j2, j3, j4 = thetas[:4]
        j0 = params[0]

        # 자동차 -> joint-0
        TC0 = np.array([ # 좌표이동 및 y축을 기준으로 90도 회전
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.166],
            [0, 0, 1, 0.099],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(j0), -np.sin(j0), 0, 0],
            [np.sin(j0),  np.cos(j0), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])

        # joint-0 -> joint-1
        ay1 = PI_HALF
        T01 = np.array([ # 좌표이동 및 y축을 기준으로 90도 회전
            [ np.cos(ay1), 0, np.sin(ay1), 0.0],
            [           0, 1,           0, 0.033],
            [-np.sin(ay1), 0, np.cos(ay1), 0.147],
            [           0, 0,           0, 1]
        ]) @ np.array([ # z축을 기준으로 j1만큼 회전
            [np.cos(j1), -np.sin(j1), 0, 0],
            [np.sin(j1),  np.cos(j1), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])
        TC1 = TC0 @ T01

        # joint-1 -> joint-2
        T12 = np.array([ # 좌표이동, 회전 없음
            [1, 0, 0, -0.155],
            [0, 1, 0,  0.0],
            [0, 0, 1,  0.0],
            [0, 0, 0,  1]
        ]) @ np.array([ # z축을 기준으로 j2만큼 회전
            [np.cos(j2), -np.sin(j2), 0, 0],
            [np.sin(j2),  np.cos(j2), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])
        TC2 = TC1 @ T12

        # joint-2 -> joint-3
        T23 = np.array([ # 좌표이동, 회전 없음
            [1, 0, 0, -0.135],
            [0, 1, 0,  0.0],
            [0, 0, 1,  0.0],
            [0, 0, 0,  1]
        ]) @ np.array([ # z축을 기준으로 j3만큼 회전
            [np.cos(j3), -np.sin(j3), 0, 0],
            [np.sin(j3),  np.cos(j3), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])
        TC3 = TC2 @ T23

        # joint-3 -> joint-4
        ay4 = -PI_HALF
        T34 = np.array([ # 좌표이동 및 y축을 기준으로 -90도 회전
            [ np.cos(ay4), 0, np.sin(ay4), -0.081],
            [           0, 1,           0,  0.0],
            [-np.sin(ay4), 0, np.cos(ay4),  0.0],
            [           0,  0,          0,  1]
        ]) @ np.array([ # z축을 기준으로 j4만큼 회전
            [np.cos(j4), -np.sin(j4), 0, 0],
            [np.sin(j4),  np.cos(j4), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])
        TC4 = TC3 @ T34

        pe_hat = TC4 @ np.array([ 0.0,   0.0,   0.123, 1])
        pc_hat = TC4 @ np.array([ 0.0,   0.0,   0.075, 1])
        oc_hat = R.from_matrix(TC4[:-1, :-1]).as_quat()

        return pe_hat[:3], np.concatenate((pc_hat[:3], oc_hat))
    
    def run_coppelia(self):
        # key input
        # Listener(on_press=self.on_press).start()
        # 시뮬레이션 실행
        # self.sim.setStepping(True)
        self.sim.startSimulation()
        index = 0

        self.goal1 = self.sim.getObject('/goalDummy')
        self.goal2 = self.sim.getObject('/bedroom')
        while self.run_flag:
            ###### path palnning & path tracking ######
            goalPos = self.sim.getObjectPosition(self.goal1, -1)
            print(f'goal position : {goalPos}')            
            # try:
            #     path = self.pathplanner.findPath(goalPos)
            #     print(f'Find the path.')
            # except:
            #     print('Fail to find path.')
            path = self.findPath(goalPos)
            if path != None:
                self.followPath(goalDummy=self.goal1, path=path)
                self.omni_wheel_control(0.0, 0.0, 0.0)
                self.clear_path()
                print('Robot reaches the goal position')
                time.sleep(2)

            ###### pick & place ######
            while self.run_flag:
                self.control_wheel_pos()
                self.control_joint()
                # print(f'index : {index}')
                if index == 1:
                    time.sleep(5)
                index += 1
                # Target Detect
                if self.target is None:
                    # 카메라 상에서 target 발견 될 때까지 실행.
                    self.target = self.targetDetector.find_target(visualize=True)

                else:
                    dist = np.linalg.norm(self.target)
                    if dist < 0.55:
                        if (self.kinematics.pick_and_place(self.target)):
                            self.target = None
                            print('escape')
                            break

                    else:
                        theta_z = np.arctan2(self.target[0], self.target[1])
                        if abs(theta_z) > 0.001:
                            self.angles[0] -= theta_z
                            self.angles[1] -= theta_z
                            self.angles[2] += theta_z
                            self.angles[3] += theta_z
                        else:
                            self.angles[0] -= 0.1
                            self.angles[1] -= 0.1
                            self.angles[2] -= 0.1
                            self.angles[3] -= 0.1
                        self.target = None
            self.init_wheelmode_velctrl()       # 주행을 위해 휠 pos. 제어 -> vel. 제어 모드 변경.
            ###### path palnning & path tracking ######
            goalPos2 = self.sim.getObjectPosition(self.goal2, -1)
            print(f'goal position : {goalPos2}')            
            path2 = self.findPath(goalPos2)
            if path2 != None:
                self.followPath(goalDummy=self.goal2, path=path2)
                self.omni_wheel_control(0.0, 0.0, 0.0)
                self.clear_path()
                print('Robot reaches the goal position')
                time.sleep(2)
                # control
                self.control_wheel()
                self.control_joint()
            
            # 시뮬레이션 step
            self.sim.step()

        # 시뮬레이션 종료
        self.sim.stopSimulation()

if __name__ == "__main__":
    client = PickAndPlace()
    client.init_coppelia()
    client.run_coppelia()