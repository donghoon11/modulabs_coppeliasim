import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from LumiBot_sensor13 import LumiBot_sensor13

class Localization(LumiBot_sensor13):
    def __init__(self):
        super().__init__()
        self.mcl = MCL()


    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.lumiBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.lumiBot_ref)[2]
        
        return x, y, theta
    
    def control_car(self):
        self.sim.setJointTargetVelocity(self.lmotor, self.v_straight)
        self.sim.setJointTargetVelocity(self.rmotor, self.v_straight)

        if self.fwd < self.fwd_cutoff:
            print('going toward the wall, turn right')
            self.sim.setJointTargetVelocity(self.lmotor, self.v_sharp)
            self.sim.setJointTargetVelocity(self.rmotor, 0)
        elif self.fwd > self.fwd_cutoff:
            if self.avg > self.max_d:
                print('going away from the wall, turn left')
                self.sim.setJointTargetVelocity(self.lmotor, self.v - self.dv)
                self.sim.setJointTargetVelocity(self.rmotor, self.v)
            elif self.avg < self.min_d:
                print('going toward the wall, turn right')
                self.sim.setJointTargetVelocity(self.lmotor, self.v)
                self.sim.setJointTargetVelocity(self.rmotor, self.v - self.dv)
            elif self.min_d < self.avg < self.max_d:
                if self.diff > self.yaw_cutoff:  # LF > LR
                    print('yaw correction: turn left')
                    self.sim.setJointTargetVelocity(self.lmotor, self.v - self.dv)
                    self.sim.setJointTargetVelocity(self.rmotor, self.v)
                elif self.diff < -self.yaw_cutoff:  # LF < LR
                    self.sim.setJointTargetVelocity(self.lmotor, self.v)
                    self.sim.setJointTargetVelocity(self.rmotor, self.v - self.dv)
    
        
    def control_feedback(self):
        flag1, LF, _, _, _ = self.sim.readProximitySensor(self.sensorLF)
        flag2, LR, _, _, _ = self.sim.readProximitySensor(self.sensorLR)
        flag3, F, _, _, _ = self.sim.readProximitySensor(self.sensorF)
        
        if flag1 == 0 and flag2 == 1:
            self.avg = LR
            self.diff = 0
        elif flag1 == 1 and flag2 == 0:
            self.avg = LF
            self.diff = 0
        elif flag1 == 1 and flag2 == 1:
            self.avg = 0.5 * (LF + LR)
            self.diff = LF - LR
        else:
            self.avg = self.avg_default
            self.diff = 0
        
        if flag3 == 1:
            self.fwd = F
        else:
            self.fwd = self.fwd_default

    
    def run_step(self, count):
        self.control_car()
        self.control_feedback()

        scan = self.read_lidars()
        loc = self.read_ref()

        # update grid
        self.mcl.update(loc, scan)


@dataclass
class Particle:
    x: float = np.random.uniform(-5, 5)
    y: float = np.random.uniform(-5,5)
    theta: float = np.random.uniform(-np.pi, np.pi)
    scan: np.array = np.full(13, 1.5)
    weight: float = 0.01

class MCL:
    def __init__(self):
        # particles : 50
        self.particles = [Particle() for i in range(50)]
        # grid
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/LumiBot/mapping.npy","rb") as f:
            self.grid = np.load(f)
        
        # plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)

        # plot object
        self.plt_objects = [None] * (15+1+13)

        # scan theta 
        self.delta = np.pi / 12
        self.scan_theta = np.array([-np.pi / 2 + self.delta * i for i in range(13)])
        self.boundary = np. pi / 2 + self.delta / 2

        # prev
        self.sigma = 1.0
        self.loc_prev = None

    def update(self, loc, scan):
        scan_vec = np.array(
            [data[1] if data[0] == 1 else 1.5 for data in scan]
        )
        if self.loc_prev:
            prev_theta = self.loc_prev[2]
            # dr = np.array(
            #     [
            #         [np.cos(-prev_theta), -np.sin(-prev_theta)],
            #         [np.sin(-prev_theta), np.cos(-prev_theta)],
            #     ]
            # ).dot(np.array([loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]]))
            dr_x, dr_y = loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]
            cos_theta = np.cos(-prev_theta)
            sin_theta = np.sin(-prev_theta)
            dr = np.array([cos_theta * dr_x - sin_theta * dr_y, 
                           sin_theta * dr_x + cos_theta * dr_y])

            dtheta = loc[2] - self.loc_prev[2]

            ### update position
            # for particle in self.particles:
            #     dp = np.array(
            #         [
            #             [np.cos(particle.theta), -np.sin(particle.theta)],
            #             [np.sin(particle.theta), np.cos(particle.theta)],                        
            #         ]
            #     ).dot(dr)
            #     particle.x += dp[0]
            #     particle.x = max(min(particle.x, 4.9), -4.9)
            #     particle.y += dp[1]
            #     particle.y = max(min(particle.y, 4.9), -4.9)
            #     particle.theta += dtheta

            #     # init scan
            #     particle.scan[:] = 1.5
            particle_x = np.array([p.x for p in self.particles])
            particle_y = np.array([p.y for p in self.particles])
            particle_theta = np.array([p.theta for p in self.particles])

            cos_p_theta = np.cos(particle_theta)
            sin_p_theta = np.sin(particle_theta)
            dx = cos_p_theta * dr[0] - sin_p_theta * dr[1]
            dy = sin_p_theta * dr[0] + cos_p_theta * dr[1]

            particle_x = np.clip(particle_x + dx, -4.9, 4.9)
            particle_y = np.clip(particle_y + dy, -4.9, 4.9)

            for i, particle in enumerate(self.particles):
                particle.x = particle_x[i]
                particle.y = particle_y[i]
                particle.theta = particle_theta[i]
                particle.scan[:] = 1.5

            # virtual scan & calc weight
            self.virtual_scan(scan_vec)
            self.resample()

        self.visualize(loc, scan)
        self.loc_prev = loc


    def virtual_scan(self, scan_vec):
        for particle in self.particles:
            # range
            dist = 1.5
            i_min = max(0, int((particle.x - dist) // 0.1 + 50))
            i_max = min(99, int((particle.x + dist) // 0.1 + 50))
            j_min = max(0, int((particle.y - dist) // 0.1 + 50))
            j_max = min(99, int((particle.y + dist) // 0.1 + 50))

            # sub grid
            sub_grid = self.grid[j_min : j_max+1, i_min : i_max+1]

            # x distance
            gx = np.arange(i_min, i_max+1) * 0.1 + 0.05 -5
            gx = np.repeat(gx.reshape(1,-1), sub_grid.shape[0], axis=0)
            dx = gx - particle.x

            # y distance
            gy = np.arange(j_min, j_max+1) * 0.1 + 0.05 -5
            gy = np.repeat(gy.reshape(1,-1).T, sub_grid.shape[1], axis=1)
            dy = gy - particle.y

            # distacne
            gd = (dx**2 + dy**2) **0.5

            # theta diff
            gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 -1)
            dtheta = gtheta - particle.theta

            while np.pi < np.max(dtheta):
                dtheta -= (np.pi < dtheta) * 2 * np.pi
            while np.min(dtheta) < -np.pi:
                dtheta += (dtheta < -np.pi) * 2 * np.pi

            # calc distance
            for i in range(13):
                area = (
                    (gd < dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i+1))
                )
                area_grid = sub_grid[area]
                area_dist = gd[area]
                assert area_grid.shape == area_dist.shape
                area_valid = area_grid > 0
                if area_valid.shape[0] > 0 and np.max(area_valid) > 0:
                    particle.scan[i] = np.min(area_dist[area_valid])
            particle.weight = 0.1 / (np.linalg.norm(scan_vec - particle.scan) + 1e-2)

    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)
        particles = np.random.choice(self.particles, len(self.particles), p=weights)
        particles = [copy.deepcopy(p) for p in particles]
     
        for particle in particles:
            particle.x += np.random.randn() * self.sigma
            particle.y += np.random.randn() * self.sigma
            particle.theta = np.random.randn() * self.sigma

        self.sigma = max(self.sigma * 0.99, 0.015)
        self.particles = particles
    
    def visualize(self, loc, scan):
        x,y, theta = loc

        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
        # robot
        (self.plt_objects[1],) = plt.plot(x, y, color="green", marker="o", markersize=10)

        # scan
        rx = x + 0.1 * np.cos(theta)
        ry = y + 0.1 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            res = res > 0
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 1.50

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        # particle
        x = [p.x  for p in self.particles]
        y = [p.y  for p in self.particles]
        c = [p.weight for p in self.particles]
        self.plt_objects[15] = plt.scatter(x, y, s=3**2, c=c, cmap="Spectral")

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = Localization()
    client.init_coppelia()
    client.run_coppelia()
