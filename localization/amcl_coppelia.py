import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
# import scipy.stats as ss

from youBot import YouBot

class LocalizationBot(YouBot):
    def __init__(self):
        super().__init__()
        self.amcl = AMCL()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        
        return x, y, theta
    
    def run_step(self):
        self.control_car()
        scan = self.read_lidars()
        loc = self.read_ref()
        # update grid
        self.amcl.update(loc, scan)


# @dataclass
# class Particle:
#     x: float = np.random.uniform(-5, 5)
#     y: float = np.random.uniform(-5, 5)
#     theta: float = np.random.uniform(-np.pi, np.pi)
#     scan: np.array = np.full(13, 2.2)
#     weight: float = 0.01


class AMCL:
    def __init__(self):
        # particles : Adaptive MCL uses a dynamic number of particles
        self.num_particles = 500
        self.min_particles = 100
        self.max_particles = 800
        # self.particles = [Particle() for i in range(self.num_particles)]
        self.particle_x = np.random.uniform(-5, 5, self.num_particles)
        self.particle_y = np.random.uniform(-5, 5, self.num_particles)
        self.particle_theta = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.particle_scan = np.full((self.num_particles, 13), 2.2)     # store scans
        self.particle_weight = np.full(self.num_particles, 0.01)

        # grid
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/localization/mapping_test.npy","rb") as f:
            self.grid = np.load(f)
        
        # plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)
        
        # plot object
        self.plt_objects = [None] * (15 + 1 + 13)  # grid, robot, scans (13), particle
        
        # scan angles
        self.delta = np.pi / 12
        self.scan_theta = np.array([-np.pi / 2 + self.delta * i for i in range(13)])
        self.boundary = np.pi / 2 + self.delta / 2
        
        # localization vars.
        self.sigma = 1.0    # noise weight
        self.loc_prev = None
        self.kld_threshold = 0.05   # KLD threshold to adapt number of particles
        self.epsilon = 0.01     # Error tolerance for KLD


    def update(self, loc, scan):
        scan_vec = np.array(
            [data[1] if data[0] == 1 else 2.2 for data in scan]
        )

        if self.loc_prev:
            prev_theta = self.loc_prev[2]
            dr_x, dr_y = loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]
            cos_theta = np.cos(-prev_theta)
            sin_theta = np.sin(-prev_theta)
            dr = np.array([cos_theta * dr_x - sin_theta * dr_y, 
                           sin_theta * dr_x + cos_theta * dr_y])
            dtheta = loc[2] - self.loc_prev[2]

            # update all particles simultaneously
            cos_p_theta = np.cos(self.particle_theta)
            sin_p_theta = np.sin(self.particle_theta)
            dx = cos_p_theta * dr[0] - sin_p_theta * dr[1]
            dy = sin_p_theta * dr[0] + cos_p_theta * dr[1]

            self.particle_x = np.clip(self.particle_x + dx, -4.9, 4.9)
            self.particle_y = np.clip(self.particle_y + dy, -4.9, 4.9)
            self.particle_theta += dtheta

            # virtual scan & calc weight
            self.virtual_scan(scan_vec)
            self.adaptive_resample()           
        self.visualize(loc, scan)
        self.loc_prev = loc
    

    def virtual_scan(self, scan_vec):
       for i in range(self.num_particles):
            # vectorized scan calculation for each particle
            particle_x = self.particle_x[i]
            particle_y = self.particle_y[i]
            particle_theta = self.particle_theta[i]

            # range
            dist = 2.25
            i_min = max(0, int((particle_x - dist) // 0.1 + 50))
            i_max = min(99, int((particle_x + dist) // 0.1 + 50))
            j_min = max(0, int((particle_y - dist) // 0.1 + 50))
            j_max = min(99, int((particle_y + dist) // 0.1 + 50))

            # sub grid
            sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]

            # x distance
            gx = np.arange(i_min, i_max + 1) * 0.1 + 0.05 - 5
            gx = np.repeat(gx.reshape(1, -1), sub_grid.shape[0], axis=0)
            dx = gx - particle_x

            # y distance
            gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
            gy = np.repeat(gy.reshape(1, -1).T, sub_grid.shape[1], axis=1)
            dy = gy - particle_y

            # distance
            gd = (dx**2 + dy**2) ** 0.5

            # theta diff
            gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 - 1)
            dtheta = gtheta - particle_theta

            dtheta = np.where(dtheta > np.pi, dtheta - 2 * np.pi, dtheta)
            dtheta = np.where(dtheta < -np.pi, dtheta + 2 * np.pi, dtheta)

            # calc distance
            for j in range(13):
                area = (
                    (gd < dist)
                    * (-self.boundary + self.delta * j <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (j + 1))
                )
                area_grid = sub_grid[area]
                area_dist = gd[area]
                assert area_grid.shape == area_dist.shape
                area_valid = area_grid > 0
               
                if area_valid.shape[0] > 0 and np.max(area_valid) > 0:
                    self.particle_scan[i,j] = np.min(area_dist[area_valid])

            error = np.linalg.norm(scan_vec - self.particle_scan[i])
            self.particle_weight[i] = np.clip(0.1 / (error + 1e-2), 1e-3, 1.0)


    def adaptive_resample(self):
        # calculate weights
        weights = self.particle_weight

        # Normalize weights so that they sum to 1
        weights /= np.sum(weights)

        # resample indices based on weights
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=weights)

        # resample particles based on the indices
        self.particle_x = self.particle_x[indices]
        self.particle_y = self.particle_y[indices]
        self.particle_theta = self.particle_theta[indices]
        self.particle_scan = self.particle_scan[indices]
        # Reset weights after resampling
        self.particle_weight = np.full(self.num_particles, 1.0 / self.num_particles)

        # calculate the KLD-sampling dynamic number of particles
        eff_particles = 1 / np.sum(weights**2)
        if eff_particles <= 0 or np.isnan(eff_particles):
            eff_particles = 1       # prevent zero or NaN values

        # calculate KLD-sampling dynamic number of particles
        kld_error = np.sqrt(self.epsilon * (1-eff_particles) / eff_particles)
        if np.isnan(kld_error) or kld_error < 0:
            kld_error = 0
        print(f'eff_particles = {eff_particles}')

        new_num_particles = min(self.max_particles, max(self.min_particles, int(self.num_particles * (1 + kld_error))))

        # adjust the number of particles
        if new_num_particles != self.num_particles:
            self.num_particles = new_num_particles
            self.particle_x = np.random.uniform(-5, 5, self.num_particles)
            self.particle_y = np.random.uniform(-5, 5, self.num_particles)
            self.particle_theta = np.random.uniform(-np.pi, np.pi, self.num_particles)
            self.particle_scan = np.full((self.num_particles, 13), 2.2)     # store scans
            self.particle_weight = np.full(self.num_particles, 1.0 / self.num_particles)

        # add noise for new particles
        self.particle_x += np.random.randn() * self.sigma
        self.particle_y += np.random.randn() * self.sigma
        self.particle_theta += np.random.randn() * self.sigma

        # ensure particles stay within bounds
        self.particle_x = np.clip(self.particle_x, -4.9, 4.9)
        self.particle_y = np.clip(self.particle_y, -4.9, 4.9)

        # gradually reduce noise
        self.sigma = max(self.sigma * 0.99, 0.015)


    def visualize(self, loc, scan):
        x, y, theta = loc
        theta += np.pi / 2
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()

        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")

        # robot
        (self.plt_objects[1],) = plt.plot(x, y, color="green", marker="o", markersize=5)
        
        # scan
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

        # particle
        x = self.particle_x
        y = self.particle_y
        c = self.particle_weight
        self.plt_objects[15] = plt.scatter(x, y, s=3**2, c=c, cmap="Spectral")

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = LocalizationBot()
    client.init_coppelia()
    client.run_coppelia()
