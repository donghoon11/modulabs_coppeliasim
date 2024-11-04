import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

import networkx as nx
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from youBotkeyboard import YouBot

class BinaryOccupancyMap(YouBot):
    def __init__(self, width, height, resolution):
        super.__init__()

        # 맵 크기와 해상도 설정
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width * resolution)
        self.grid_height = int(height * resolution)
        
        # 점유 상태 그리드 초기화 (0: 빈 공간, 1: 장애물)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
    
    def world_to_grid(self, points):
        # 실수 좌표를 그리드 좌표로 변환
        grid_points = np.floor(points * self.resolution).astype(int)
        grid_points[:, 0] = np.clip(grid_points[:, 0], 0, self.grid_width - 1)
        grid_points[:, 1] = np.clip(grid_points[:, 1], 0, self.grid_height - 1)
        return grid_points

    def set_occupancy(self, points, values):
        # 점유 상태 설정
        grid_points = self.world_to_grid(points)
        for (x, y), value in zip(grid_points, values):
            self.grid[y, x] = 1 if value > 0 else 0

    def get_occupancy(self, points):
        # 점유 상태 조회
        grid_points = self.world_to_grid(points)
        return [self.grid[y, x] for x, y in grid_points]

    def inflate(self, radius):
        # 맵 팽창: 주어진 반경만큼 장애물 영역을 확장
        grid_radius = int(radius * self.resolution)
        self.grid = binary_dilation(self.grid, structure=np.ones((2 * grid_radius + 1, 2 * grid_radius + 1))).astype(np.uint8)

    def show(self, ax=None):
        # 맵 시각화
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.grid, origin='lower', cmap='gray_r', extent=(0, self.width, 0, self.height))
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Binary Occupancy Map')
        plt.show()

    def run_coppelia(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            self.set_occupancy(points, np.ones(len(points)))
            fig, ax = plt.subplots()
            self.show(ax)

            robot_radius = 0.5
            self.inflate(robot_radius)
            fig, ax = plt.subplots()
            self.show(ax)
        self.sim.stopSimulation()

if __name__ == "__main__":
    map = BinaryOccupancyMap()
    map.init_coppelia()
    map.run_coppelia()


# if __name__ == "__main__":
#     # 사용 예시
#     map = BinaryOccupancyMap(10, 10, 2)
#     x = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
#     y = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5])
#     points = np.column_stack((x, y))

#     map.set_occupancy(points, np.ones(len(points)))
#     fig, ax = plt.subplots()
#     map.show(ax)

#     # 팽창 예시
#     robot_radius = 0.5
#     map.inflate(robot_radius)
#     fig, ax = plt.subplots()
#     map.show(ax)


class MobileRobotPRM(BinaryOccupancyMap):
    def __init__(self, num_nodes=100, connection_distance=5):
        super().__init__()
        self.map = BinaryOccupancyMap(10,10,2)
        self.map
        self.num_nodes = num_nodes
        self.connection_distance = connection_distance
        self.nodes = []
        self.graph = nx.Graph()

    def is_free(self, point):
        symbol = True if self.map.get_occupancy([point])[0] == 0 else False
        return symbol
    
    def generate_random_nodes(self):
        # 맵 내에서 무작위 노드를 생성하고, 장애물이 없는 노드만 추가
        while len(self.nodes) < self.num_nodes:
            x = np.random.uniform(0, self.map.width)
            y = np.random.uniform(0, self.map.height)
            point = (x, y)
            if self.is_free(point):
                self.nodes.append(point)
                self.graph.add_node(len(self.nodes) - 1, pos = point)

    def connect_nodes(self):
        # KDTree를 사용해 각 노드를 인접한 노드와 연결
        kdtree = KDTree(self.nodes)
        for i, node in enumerate(self.nodes):
            indices = kdtree.query_ball_point(node, self.connection_distance)
            for j in indices:
                if i != j:
                    if not self.is_path_obstructed(self.nodes[i], self.nodes[j]):
                        self.graph.add_edge(i, j, weight=np.linalg.norm(np.array(node) - np.array(self.nodes[j])))

    def is_path_obstructed(self, point1, point2):
        # 두 포인트 사이에 장애물이 있는지 확인
        num_points = int(np.linalg.norm(np.array(point2) - np.array(point1)) * self.map.resolution)
        points = np.linspace(point1, point2, num_points)
        for point in points:
            if not self.is_free(point):
                return True
        return False
    
    def find_path(self, start, goal):
        # 시작점과 목표점에서 가장 가까운 노드를 찾고 최단 경로를 탐색
        self.generate_random_nodes()
        self.connect_nodes()
        
        # 시작점과 목표점의 인접 노드 찾기
        self.nodes.append(start)
        self.nodes.append(goal)
        start_idx = len(self.nodes) - 2
        goal_idx = len(self.nodes) - 1
        self.graph.add_node(start_idx, pos=start)
        self.graph.add_node(goal_idx, pos=goal)

        # 시작, 목표 위치에서 연결 가능한 인접 노드 연결
        neighbors = NearestNeighbors(radius=self.connection_distance).fit(self.nodes)
        for idx, location in [(start_idx, start), (goal_idx, goal)]:
            distances, indices = neighbors.radius_neighbors([location], return_distance=True)
            for i in indices[0]:
                if i != idx and not self.is_path_obstructed(location, self.nodes[i]):
                    self.graph.add_edge(idx, i, weight=np.linalg.norm(np.array(location) - np.array(self.nodes[i])))
        
        try:
            path = nx.shortest_path(self.graph, source=start_idx, target=goal_idx, weight='weight')
            return [self.nodes[i] for i in path]
        except nx.NetworkXNoPath:
            print("No path found between start and goal.")
            return None

    def show(self, path=None):
        # 맵, 노드, 엣지 및 경로를 시각화
        fig, ax = plt.subplots()
        self.map.show(ax=ax)

        # 노드와 엣지 표시
        for (i, j) in self.graph.edges:
            pos_i = self.graph.nodes[i]['pos']
            pos_j = self.graph.nodes[j]['pos']
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'bo-', linewidth=0.5, markersize=2)

        # 경로 표시
        if path is not None:
            path_points = np.array(path)
            ax.plot(path_points[:, 0], path_points[:, 1], 'ro-', linewidth=2)

        ax.set_title("Probabilistic Roadmap (PRM)")
        plt.show()

if __name__ == "__main__":

    # 맵 생성
    occupancy_map = BinaryOccupancyMap(10, 10, 2)
    x = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
    y = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    points = np.column_stack((x, y))

    occupancy_map.set_occupancy(points, np.ones(len(points)))

    # PRM 생성 및 경로 찾기
    prm = MobileRobotPRM(occupancy_map, num_nodes=100, connection_distance=2)
    start = (1, 1)
    goal = (9, 9)
    path = prm.find_path(start, goal)
    prm.show(path=path)

