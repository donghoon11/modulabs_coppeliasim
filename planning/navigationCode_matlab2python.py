
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

def calcMatrix():
    # 파라미터 설정
    r = 0.05
    l = 0.2792

    # 각도 설정
    a = [32.5, 147.5, -147.5, -32.5]
    b = [57.5, -57.5, -122.5, 122.5]
    g = [45, 135, 45, 135]

    # k 계산
    k = [a[i] + b[i] + g[i] for i in range(4)]

    # A와 B 행렬 정의
    A = np.array([
        [np.sin(np.radians(k[0])), -np.cos(np.radians(k[0])), -l * np.cos(np.radians(b[0] + g[0]))],
        [np.sin(np.radians(k[1])), -np.cos(np.radians(k[1])), -l * np.cos(np.radians(b[1] + g[1]))],
        [np.sin(np.radians(k[2])), -np.cos(np.radians(k[2])), -l * np.cos(np.radians(b[2] + g[2]))],
        [np.sin(np.radians(k[3])), -np.cos(np.radians(k[3])), -l * np.cos(np.radians(b[3] + g[3]))]
    ])

    B = np.diag([r * np.cos(np.radians(g[i])) for i in range(4)])
    
    return A, B

class Nav2:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

    def init_coppelia(self):
        # 객체 핸들 정의
        robot = Shape('youBot')
        wheels = [Joint('rollingJoint_rr'), Joint('rollingJoint_fr'), Joint('rollingJoint_rl'), Joint('rollingJoint_fl')]

        # 초기 속도 설정
        Vx, Vy, thetadot = 0.05, 0.05, 0
        W = np.array([0, 0, 0, 0])

        # 웨이포인트와 목표 위치
        waypoints = [(x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)]
        waypointNumber = len(waypoints)
        d = 0.05  # 데드존

        try:
            while True:
                positionRobot = robot.get_position()
                robotX, robotY = positionRobot[0], positionRobot[1]
                robotXrounded, robotYrounded = round(robotX, 2), round(robotY, 2)

                # 로봇 속도 벡터 V 및 휠 속도 벡터 W 계산
                V = np.array([Vx, Vy, thetadot])
                Y = np.linalg.inv(B)
                W = Y @ A @ V

                # 목표점에 도달하면 속도 업데이트
                for i, (x_goal, y_goal) in enumerate(waypoints, start=2):
                    if (robotXrounded < x_goal + d) and (robotXrounded > x_goal - d) and (robotYrounded < y_goal + d) and (robotYrounded > y_goal - d):
                        Vx, Vy = 0.07 * locals().get(f'vely_{i}', 0), 0.07 * locals().get(f'velx_{i}', 0)
                        if i == waypointNumber:
                            Vx, Vy = 0, 0  # 마지막 목표점 도달 시 정지

                # 휠 속도 설정
                for i, wheel in enumerate(wheels):
                    wheel.set_velocity(W[i])

                pr.step()  # 한 스텝 진행

        except KeyboardInterrupt:
            print("Simulation ended")

        finally:
            pr.stop()
            pr.shutdown()
