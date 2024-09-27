import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from youBot import YouBot


class LocalizationBot(YouBot):
    def __init__(self):
        super().__init__()
        self.ekf = EKF()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta

    def run_step(self, count):
        self.control_car()
        scan = self.read_lidars()
        loc = self.read_ref()
        self.ekf.update(loc, scan)


@dataclass
class EKFState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    P: np.array = np.eye(3) * 0.1  # 초기 공분산 행렬


class EKF:
    def __init__(self):
        # 초기 상태
        self.state = EKFState()

        # 상태 천이 모델 잡음 공분산
        self.Q = np.diag([0.1, 0.1, np.deg2rad(1)]) ** 2

        # 센서 잡음 공분산 (라이다 등)
        self.R = np.diag([0.2] * 13) ** 2

        # 라이다 각도
        self.delta = np.pi / 12
        self.scan_theta = np.array([-np.pi / 2 + self.delta * i for i in range(13)])

        # 맵 (grid)
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/localization/mapping_test.npy", "rb") as f:
            self.grid = np.load(f)

        # 시각화 설정
        self.plt_objects = [None] * (15 + 1 + 13)
        
        # plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.plot_R, self.plot_P = np.meshgrid(r, p)


    def predict(self, v, omega, dt):
        # 상태 예측
        x, y, theta = self.state.x, self.state.y, self.state.theta

        if abs(omega) > 1e-6:
            # 비선형 상태 천이 모델
            x_pred = x - v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * dt)
            y_pred = y + v / omega * np.cos(theta) - v / omega * np.cos(theta + omega * dt)
        else:
            # 직선 운동일 때
            x_pred = x + v * np.cos(theta) * dt
            y_pred = y + v * np.sin(theta) * dt
        theta_pred = theta + omega * dt

        # 자코비안 행렬 (상태 천이 모델의 선형화)
        F = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1, v * dt * np.cos(theta)],
            [0, 0, 1]
        ])

        # 공분산 예측
        self.state.P = np.dot(F, np.dot(self.state.P, F.T)) + self.Q

        # 상태 업데이트
        self.state.x, self.state.y, self.state.theta = x_pred, y_pred, theta_pred


    def update(self, loc, scan):
        # 예측 단계에서 이동 거리를 계산 (단순히 임의로 설정된 값 사용)
        v = 0.1
        omega = 0.1
        dt = 0.1
        self.predict(v, omega, dt)

        # 라이다 센서 데이터
        scan_vec = np.array([data[1] if data[0] == 1 else 2.2 for data in scan])

        # 측정 예측 (현재 상태에서 예상되는 라이다 값 계산)
        z_pred = self.virtual_scan(self.state.x, self.state.y, self.state.theta)

        # 칼만 이득 계산
        H = self.calculate_jacobian(self.state.x, self.state.y, self.state.theta)
        S = np.dot(H, np.dot(self.state.P, H.T)) + self.R
        K = np.dot(self.state.P, np.dot(H.T, np.linalg.inv(S)))

        # 상태 업데이트
        y = scan_vec - z_pred
        dx = np.dot(K, y)
        self.state.x += dx[0]
        self.state.y += dx[1]
        self.state.theta += dx[2]
        self.state.P = np.dot(np.eye(3) - np.dot(K, H), self.state.P)

        # 시각화
        self.visualize(loc, scan)


    def calculate_jacobian(self, x, y, theta):
        # 자코비안 행렬 계산 (센서 모델의 선형화)
        H = np.zeros((13, 3))
        for i in range(13):
            angle = theta + self.scan_theta[i]
            H[i, 0] = np.cos(angle)
            H[i, 1] = np.sin(angle)
            H[i, 2] = -np.sin(angle) * (x * np.cos(angle) + y * np.sin(angle))
        return H


    def virtual_scan(self, x, y, theta):
        # 로봇의 위치에서 예상되는 라이다 측정값 (맵을 바탕으로)
        scan = np.full(13, 2.2)
        dist = 2.25

        i_min = max(0, int((x - dist) // 0.1 + 50))
        i_max = min(99, int((x + dist) // 0.1 + 50))
        j_min = max(0, int((y - dist) // 0.1 + 50))
        j_max = min(99, int((y + dist) // 0.1 + 50))

        sub_grid = self.grid[j_min: j_max + 1, i_min: i_max + 1]

        gx = np.arange(i_min, i_max + 1) * 0.1 + 0.05 - 5
        gx = np.repeat(gx.reshape(1, -1), sub_grid.shape[0], axis=0)
        dx = gx - x

        gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
        gy = np.repeat(gy.reshape(1, -1).T, sub_grid.shape[1], axis=1)
        dy = gy - y

        gd = (dx ** 2 + dy ** 2) ** 0.5
        gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 - 1)
        dtheta = gtheta - theta

        if dtheta.size > 0:
            while np.pi < np.max(dtheta):
                dtheta -= (np.pi < dtheta) * 2 * np.pi
            while np.min(dtheta) < -np.pi:
                dtheta += (dtheta < -np.pi) * 2 * np.pi
        else:
            print("dtheta is an empty array")

        for i in range(13):
            area = (gd < dist) * (-self.scan_theta[i] <= dtheta) * (dtheta <= self.scan_theta[i])
            area_grid = sub_grid[area]
            area_dist = gd[area]
            area_valid = area_grid > 0
            if area_valid.shape[0] > 0 and np.max(area_valid) > 0:
                scan[i] = np.min(area_dist[area_valid])
        return scan


    def visualize(self, loc, scan):
        x, y, theta = loc

        # Clear object
        for object in self.plt_objects:
            if object:
                object.remove()

        # Grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.plot_R, self.plot_P, grid, cmap="gray")

        # Robot
        (self.plt_objects[1],) = plt.plot(x, y, color="green", marker="o", markersize=5)

        # Scan
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            res = res > 0
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.pause(0.01)


if __name__ == "__main__":
    client = LocalizationBot()
    client.init_coppelia()
    client.run_coppelia()