# # Copyright 2024 @with-RL
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# #     http://www.apache.org/licenses/LICENSE-2.0

# from dataclasses import dataclass

# import numpy as np
# import matplotlib.pyplot as plt

# from youBot import YouBot


# class Mapping(YouBot):
#     def __init__(self):
#         super().__init__()
#         self.grid = Grid()

#     def read_ref(self):
#         position = self.sim.getObjectPosition(self.youBot_ref)
#         orientation = self.sim.getObjectOrientation(self.youBot_ref)
#         return position + orientation

#     def run_step(self, count):
#         # car control
#         self.control_car()
#         # arm control
#         self.control_arm()
#         # arm gripper
#         self.control_gripper()
#         # read position and orientation
#         loc = self.read_ref()
#         # update grid
#         self.grid.update(loc)


# @dataclass
# class LidarInfo:
#     offset = 0.275  # distance from youBot_ref
#     alpha = 2.2  # scan max distance
#     beta = np.pi / 12  # scan angle
#     scan_count = 13  # scan count
#     scan_theta = np.array([-np.pi / 2 + (np.pi / 12) * i for i in range(13)])
#     scan_cos_min = np.cos(-np.pi / 2 - (np.pi / 12) / 2)


# class Grid:
#     def __init__(self):
#         self.lidar_info = LidarInfo()
#         # grid data
#         self.grid = np.zeros((100, 100, 3))  # x, y, occupy
#         self.grid[:, :, 0] = np.linspace(-4.95, 4.95, 100).reshape(1, 100)
#         self.grid[:, :, 1] = np.linspace(-4.95, 4.95, 100).reshape(100, 1)
#         # plot grid
#         r = np.linspace(-5, 5, 101)
#         p = np.linspace(-5, 5, 101)
#         self.R, self.P = np.meshgrid(r, p)
#         # plot object
#         self.plt_objects = [None] * 3  # grid, robot, head

#     def update(self, loc):
#         self.mapping(loc)
#         self.visualize(loc)

#     def mapping(self, loc):
#         x, y, z, theta_x, theta_y, theta_z = loc
#         # scanner position
#         rx = x + self.lidar_info.offset * np.cos(theta_z)
#         ry = y + self.lidar_info.offset * np.sin(theta_z)
#         scan_position = np.array([rx, ry])
#         # position of grid (relative position)
#         grid_xy = self.grid[:, :, :2] - scan_position.reshape(1, 1, -1)
#         # distance of grid
#         distance = np.linalg.norm(grid_xy, axis=-1)
#         # angle of grid
#         unit_xy = grid_xy / np.linalg.norm(grid_xy, axis=-1, keepdims=True)
#         unit_angle = np.array([np.cos(theta_z), np.sin(theta_z)]).reshape(2, 1)
#         angle = np.matmul(unit_xy, unit_angle).reshape(100, 100)
#         # check valid
#         valid = (distance <= self.lidar_info.alpha) * (
#             angle >= self.lidar_info.scan_cos_min
#         )
#         self.grid[:, :, 2] = valid.astype(np.float64)

#     def visualize(self, loc):
#         x, y, z, theta_x, theta_y, theta_z = loc
#         # clear object
#         for object in self.plt_objects:
#             if object:
#                 object.remove()
#         # grid
#         grid = -self.grid[:, :, 2]
#         self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
#         # robot
#         (self.plt_objects[1],) = plt.plot(
#             x, y, color="green", marker="o", markersize=10
#         )
#         # head
#         xi = x + self.lidar_info.alpha * np.cos(theta_z)
#         yi = y + self.lidar_info.alpha * np.sin(theta_z)
#         (self.plt_objects[2],) = plt.plot([x, xi], [y, yi], "--b")

#         plt.xlim(-5, 5)
#         plt.ylim(-5, 5)
#         plt.gca().set_aspect("equal")
#         plt.pause(0.001)


# if __name__ == "__main__":
#     client = Mapping()
#     client.init_coppelia()
#     client.run_coppelia()



#########################################################################################
import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot
from dataclasses import dataclass

class Mapping(YouBot):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        """로봇의 2D 위치(x, y)와 방향(theta)을 반환"""
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta

    def run_step(self, count):
        """각 스텝에서 라이다 데이터와 로봇의 위치를 읽어 그리드를 업데이트"""
        scan = self.read_lidars()  # 라이다 데이터를 읽어옴
        loc = self.read_ref()  # 로봇의 위치 정보 읽어옴
        self.grid.update(loc, scan)  # 그리드 업데이트

@dataclass
class LidarInfo:
    offset: float = 0.1  # 로봇에서 라이다 센서까지의 거리
    alpha: float = 2.0  # 라이다 스캔 최대 거리
    scan_count: int = 13  # 스캔 라이다 수
    delta: float = np.pi / 12  # 각도 간격
    scan_theta: np.ndarray = np.array([-np.pi / 2 + (np.pi / 12) * i for i in range(13)])  # 스캔 각도 배열

class Grid:
    def __init__(self):
        """그리드 초기화 및 그리드 시각화 설정"""
        self.lidar_info = LidarInfo()
        self.grid = np.zeros((100, 100, 3))  # 그리드: (x, y, occupy)
        self.grid[:, :, 0] = np.linspace(-4.95, 4.95, 100).reshape(1, 100)
        self.grid[:, :, 1] = np.linspace(-4.95, 4.95, 100).reshape(100, 1)

        # 그리드 좌표 시각화용 설정
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)

        self.plt_objects = [None] * 3  # grid, robot, head

    def update(self, loc, scan):
        """로봇의 위치와 라이다 스캔 데이터를 이용해 그리드 업데이트 및 시각화"""
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    def mapping(self, loc, scan):
        """로봇 위치와 라이다 스캔 데이터를 이용해 그리드 맵 업데이트"""
        x, y, theta = loc
        rx = x + self.lidar_info.offset * np.cos(theta)
        ry = y + self.lidar_info.offset * np.sin(theta)
        scan_position = np.array([rx, ry])

        # 그리드 내 모든 포인트와 로봇의 상대 거리 및 각도 계산
        grid_xy = self.grid[:, :, :2] - scan_position.reshape(1, 1, -1)
        distance = np.linalg.norm(grid_xy, axis=-1)
        unit_xy = grid_xy / np.linalg.norm(grid_xy, axis=-1, keepdims=True)
        angle = np.matmul(unit_xy, np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)).reshape(100, 100)

        # 라이다 스캔 결과 반영
        for i, (res, dist, _, _, _) in enumerate(scan):
            scan_angle = theta + self.lidar_info.scan_theta[i]
            scan_dist = min(self.lidar_info.alpha, dist if res else self.lidar_info.alpha)
            xi = rx + scan_dist * np.cos(scan_angle)
            yi = ry + scan_dist * np.sin(scan_angle)

            # 장애물 존재 여부에 따른 맵 업데이트
            sub_grid = self.grid[
                (distance <= scan_dist) & (angle >= np.cos(scan_angle - self.lidar_info.delta / 2))
                & (angle <= np.cos(scan_angle + self.lidar_info.delta / 2))
            ]
            if res:
                sub_grid[:, :, 2] += 0.5  # 장애물 발견
            else:
                sub_grid[:, :, 2] -= 0.5  # 자유 공간

        np.clip(self.grid[:, :, 2], -5, 5, out=self.grid[:, :, 2])

    def save(self):
        """그리드 데이터를 파일에 저장"""
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/localization/youbot_mapping.npy", "wb") as f:
            np.save(f, self.grid)

    def visualize(self, loc, scan):
        """로봇의 위치 및 라이다 스캔 데이터를 이용한 시각화"""
        x, y, theta = loc

        # 이전 시각화된 객체 제거
        for object in self.plt_objects:
            if object:
                object.remove()

        # 그리드 시각화
        grid = -self.grid[:, :, 2]
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")

        # 로봇 시각화
        (self.plt_objects[1],) = plt.plot(x, y, color="green", marker="o", markersize=10)

        # 라이다 스캔 시각화
        rx = x + self.lidar_info.offset * np.cos(theta)
        ry = y + self.lidar_info.offset * np.sin(theta)
        for i, (res, dist, _, _, _) in enumerate(scan):
            style = "--r" if res else "--b"
            dist = dist if res else self.lidar_info.alpha
            scan_angle = theta + self.lidar_info.scan_theta[i]
            xi = rx + dist * np.cos(scan_angle)
            yi = ry + dist * np.sin(scan_angle)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = Mapping()
    client.init_coppelia()
    client.run_coppelia()
