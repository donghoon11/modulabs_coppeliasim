# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot


class LocalizationBot(YouBot):
    def __init__(self):
        super().__init__()
        self.ekf = ExtendedKalmanFilterLocalization()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta

    def run_step(self, count):
        # car control
        self.control_car()
        # read lidars
        scan = self.read_lidars()
        # read youBot_ref
        loc = self.read_ref()
        # update EKF
        self.ekf.update(loc, scan)


class ExtendedKalmanFilterLocalization:
    def __init__(self):
        # 상태 변수: x, y, theta
        self.state = np.array([0.0, 0.0, 0.0])  # 초기 상태 [x, y, theta]
        self.covariance = np.eye(3) * 0.1  # 초기 공분산 행렬

        # 프로세스 노이즈 및 측정 노이즈
        self.process_noise = np.eye(3) * 0.01  # 시스템 노이즈 공분산
        self.measurement_noise = np.eye(13) * 0.1  # 측정 노이즈 공분산

        # 스캐너와 각도 설정
        self.scan_theta = np.array([-np.pi / 2 + np.pi / 12 * i for i in range(13)])
        self.boundary = np.pi / 2 + np.pi / 24

        # 맵 불러오기
        with open("/home/oh/my_coppeliasim/coppelia-practice/youBot/mapping.npy", "rb") as f:
            self.grid = np.load(f)

        # 시각화 설정
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)
        self.plt_objects = [None] * (15 + 1 + 13)  # grid, robot, scans (13)

        self.loc_prev = None

    def predict(self, control):
        """
        예측 단계: 시스템 모델을 사용하여 상태를 예측.
        """
        x, y, theta = self.state
        delta_x, delta_y, delta_theta = control

        # 상태 갱신 (비선형 모델)
        theta_new = theta + delta_theta
        x_new = x + delta_x * np.cos(theta)
        y_new = y + delta_x * np.sin(theta)

        # 상태 업데이트
        self.state = np.array([x_new, y_new, theta_new])

        # Jacobian 행렬(F)로 공분산 예측
        F = np.array([[1, 0, -delta_x * np.sin(theta)],
                      [0, 1, delta_x * np.cos(theta)],
                      [0, 0, 1]])

        # 공분산 갱신
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, loc, scan):
        """
        EKF 업데이트 단계: 센서 데이터(라이다 스캔)를 사용하여 상태 업데이트.
        """
        scan_vec = np.array([data[1] if data[0] == 1 else 2.2 for _, data in enumerate(scan)])

        if self.loc_prev:
            prev_theta = self.loc_prev[2]
            dr = np.array([[np.cos(-prev_theta), -np.sin(-prev_theta)],
                           [np.sin(-prev_theta), np.cos(-prev_theta)]]
                          ).dot(np.array([loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]]))
            dtheta = loc[2] - self.loc_prev[2]

            control = [dr[0], dr[1], dtheta]
            self.predict(control)

            self.measurement_update(scan_vec)

        self.visualize(loc, scan)
        self.loc_prev = loc

    def measurement_update(self, scan_vec):
        """
        측정 업데이트: 라이다 스캔 결과와 예측 상태 비교하여 상태 갱신.
        """
        H, h_scan = self.compute_jacobian_and_scan(self.state)

        # 칼만 이득(K)
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # 상태 업데이트
        self.state = self.state + K @ (scan_vec - h_scan)

        # 공분산 갱신
        self.covariance = (np.eye(3) - K @ H) @ self.covariance

    def compute_jacobian_and_scan(self, state):
        """
        상태에 따른 Jacobian 행렬(H) 및 예상 라이다 스캔(h(x)) 계산.
        """
        x, y, theta = state

        # 예상 라이다 스캔 값 계산 (h(x))
        h_scan = np.full(13, 2.2)
        for i in range(13):
            dist = 2.25  # 라이다 범위
            for d in np.linspace(0, dist, 100):
                x_scan = x + d * np.cos(theta + self.scan_theta[i])
                y_scan = y + d * np.sin(theta + self.scan_theta[i])

                # 맵 경계를 넘지 않는지 확인
                if -5 <= x_scan <= 5 and -5 <= y_scan <= 5:
                    i_grid = int((x_scan + 5) * 10)
                    j_grid = int((y_scan + 5) * 10)
                    if self.grid[j_grid, i_grid] > 0:  # 장애물 발견
                        h_scan[i] = d
                        break

        # Jacobian 행렬(H) 계산
        H = np.zeros((13, 3))
        for i in range(13):
            H[i, 0] = -np.cos(theta + self.scan_theta[i])
            H[i, 1] = -np.sin(theta + self.scan_theta[i])
            H[i, 2] = (x - h_scan[i]) * np.sin(theta + self.scan_theta[i]) - (y - h_scan[i]) * np.cos(theta + self.scan_theta[i])

        return H, h_scan


    def visualize(self, loc, scan):
        """
        로봇의 실제 위치(loc)와 추정된 위치를 시각화하고,
        라이다 스캔 데이터도 함께 표시합니다.
        """
        x, y, theta = loc
        est_x, est_y, est_theta = self.state

        # clear previous plots
        for obj in self.plt_objects:
            if obj:
                obj.remove()

        # 맵 시각화 (그리드)
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")

        # 실제 로봇의 위치 시각화
        self.plt_objects[1], = plt.plot(x, y, color="green", marker="o", markersize=10, label="True Position")
        self.plt_objects[2] = plt.arrow(x, y, 0.5 * np.cos(theta), 0.5 * np.sin(theta), color="green", head_width=0.1)

        # 추정된 로봇의 위치 시각화
        self.plt_objects[3], = plt.plot(est_x, est_y, color="red", marker="x", markersize=10, label="Estimated Position")
        self.plt_objects[4] = plt.arrow(est_x, est_y, 0.5 * np.cos(est_theta), 0.5 * np.sin(est_theta), color="red", head_width=0.1)

        # 라이다 스캔 시각화
        rx = est_x + 0.275 * np.cos(est_theta)
        ry = est_y + 0.275 * np.sin(est_theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            res = res > 0
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = est_theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            self.plt_objects[5 + i], = plt.plot([rx, xi], [ry, yi], style)

        # 그래프 설정
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.pause(0.001)


if __name__ == "__main__":
    client = LocalizationBot()
    client.init_coppelia()
    client.run_coppelia()
