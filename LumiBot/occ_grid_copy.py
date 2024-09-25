import numpy as np
import matplotlib.pyplot as plt

from LumiBot_sensor13 import LumiBot_sensor13 

class MappingBot(LumiBot_sensor13):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.lumiBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.lumiBot_ref)[2]
        
        return x, y, theta
    
    
    def run_step(self, count):      # 파라미터로 count 넣지 않으면 부모 클래스에 의해 오류 발생.
        scan = self.read_lidars()
        loc = self.read_ref()

        # update grid
        self.grid.update(loc, scan)


class Grid():
    def __init__(self):
        ### set grid size
        # self.grid = np.zeros((100,100))      # 2d array
        self.grid = np.zeros((100,100,3))
        self.grid[:,:,0] = np.linspace(-4.95, 4.95, 100).reshape(1,100)
        self.grid[:,:,1] = np.linspace(-4.95, 4.95, 100).reshape(100,1)

        ### plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)

        ### plot object
        # self.plt_objects = [None] * 15
        self.plt_objects = [None] * 3   # grid, robot, head

        ### scan theta
        self.delta = np.pi / 12

        ### -pi/2(ref) ->  set the scan angle for each lidars
        self.scan_theta = np.array([-np.pi/2 + self.delta * i for i in range(13)])

        ### set boundary angle
        self.boundary = np.pi / 2 + self.delta / 2

        ### min distance 
        self.min_dist = (2* (0.05**2)) ** 0.5       # 0.07071


    def update(self, loc, scan):
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    
    def mapping(self, loc, scan):
        x, y, theta = loc
        
        # set variables for inverse sensor model.
        rx = x + 0.1 * np.cos(theta)        # 1D array
        ry = y + 0.1 * np.sin(theta)        # 1D array
        dist = 2.0

        # xy_resolution = 0.1*0.1
        i_min = max(0, int((rx - dist) // 0.1 + 50))
        i_max = min(99, int((rx - dist) // 0.1 + 50))
        j_min = max(0, int((ry - dist) // 0.1 + 50))
        j_max = min(99, int((ry - dist) // 0.1 + 50))
        
        # sub grid (row, col = y, x)
        sub_grid = self.grid[j_min:j_max+1, i_min:i_max+1]

        ### set params for visualizaing the map
        # x distance 
        gx = np.arange(i_min, i_max+1) * 0.1 + 0.05 - 5
        gx = np.repeat(gx.reshape(1,-1), sub_grid.shape[0], axis=0)
        dx = gx - rx

        # y distance
        gy = np.arange(j_min, j_max+1) * 0.1 + 0.05 - 5
        gy = np.repeat(gy.reshape(1,-1), sub_grid.shape[1], axis=1)
        dy = gy - ry
        
        # distance
        gd = (dx**2 + dy**2) ** 0.5

        # theta diff
        gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 -1)
        dtheta = gtheta - theta

        # set the range of angles (-pi ~ pi)
        while np.pi < np.max(dtheta):
            dtheta -= (np.pi < dtheta) * 2 * np.pi      # multiple pi means 'trans. [deg] -> [rad];.
        while np.min(dtheta) < -np.pi :
            dtheta += (dtheta < -np.pi) * 2 * np.pi

        # inverse sensor model
        # res : bool that detect an obstacle
        # dist : distancae between the robot and an obstacle
        for i in range(13):
            res, dist, _, _ , _ = scan[i]
            # if not detect an obstacle,
            if res == 0:
                area = (
                    (gd <= 2.0) * (-self.boundary + self.delta * i <= dtheta) * (dtheta <= -self.boundary + self.delta * (i+1))
                )
                sub_grid[area] -= 0.5
            else:
                dist = min(2.0, dist)
                detect_area = (
                    (np.abs(gd - dist) < self.min_dist) * (-self.boundary + self.delta * i <= dtheta) * (dtheta <= -self.boundary + self.delta * (i+1))
                )
                sub_grid[detect_area] += 0.5
                free_area = (
                    (gd <= dist - self.min_dist) * (-self.boundary + self.delta * i <= dtheta) * (dtheta <= -self.boundary + self.delta * (i+1))
                )
                sub_grid[free_area] -= 0.5
        np.clip(self.grid, -5, 5, out=self.grid)

    
    def save(self):
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/LumiBot/mapping.npy", "wb") as f:
            np.save(f, self.grid)


    def visualize(self, loc, scan):
        x, y, theta = loc
        for object in self.plt_objects:
            if object:
                object.remove()

        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap='gray')
        
        # robot
        (self.plt_objects[1], ) = plt.plot(x,y,color="green", marker = "o", markersize=10)

        # scan
        rx = x + 0.1 * np.cos(theta)
        ry = y + 0.1 * np.sin(theta)

        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 1.5

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = MappingBot()
    client.init_coppelia()
    client.run_coppelia()