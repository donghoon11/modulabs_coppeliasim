# amcl_coppelia.py legacy

import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from youBot import YouBot

from scipy.spatial import KDTree


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
        ### update grid
        # self.amcl.update(loc, scan)
        # v, omega = 0.5, 0.1
        # dt = 0.1
        # self.amcl.rk4_update(dt, v, omega, loc, scan)

        self.amcl.icp_update(loc, scan)


@dataclass
class Particle:
    x: float = np.random.uniform(-5, 5)
    y: float = np.random.uniform(-5, 5)
    theta: float = np.random.uniform(-np.pi, np.pi)
    scan: np.array = np.full(13, 2.2)
    weight: float = 0.01


class AMCL:
    def __init__(self):
        # particles : Adaptive MCL uses a dynamic number of particles
        self.num_particles = 800
        self.min_particles = 500
        self.max_particles = 1000
        self.particles = [Particle() for i in range(self.num_particles)]

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

    
    def icp(self, scan_data, virtual_scan_data, max_iteration = 20, threshold=1e-4):
        """
        Iterative Closet Poitn algorithm to registrate points (to align virutal scan and real scan)
        : return : 최적의 변환 행렬
        """
        prev_error = float('inf')
        for i in range(max_iteration):
            # closet point matching using KDTree
            tree =KDTree(virtual_scan_data)
            distances, indices = tree.query(scan_data)

            # select the corresponding poitns in virtual_scan
            corr_points = virtual_scan_data[indices]
            
            # compute the centroids
            scan_centroid = np.mean(scan_data, axis=0)
            corr_centroid = np.mean(corr_points, axis=0)

            # center the points around the centroids
            scan_center = scan_data - scan_centroid
            corr_center = corr_points - corr_centroid

            # SVD to compute the rot.
            H = np.dot(scan_center.T, corr_center)
            U, _, V_T = np.linalg.svd(H)
            R = np.dot(V_T.T, U.T)

            if np.linalg.det(R) < 0:
                V_T[-1, :] *= -1
                R = np.dot(V_T.T, U.T)

            t = corr_centroid.T - np.dot(R, scan_centroid.T)
            scan_data_trans = np.dot(scan_data, R.T) + t.T

            # optimization
            error = np.mean(distances)
            if abs(prev_error - error) < threshold:
                break
            prev_error = error

        return R, t


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

            particle_x = np.array([p.x for p in self.particles])
            particle_y = np.array([p.y for p in self.particles])
            particle_theta = np.array([p.theta for p in self.particles])

            cos_p_theta = np.cos(particle_theta)
            sin_p_theta = np.sin(particle_theta)
            dx = cos_p_theta * dr[0] - sin_p_theta * dr[1]
            dy = sin_p_theta * dr[0] + cos_p_theta * dr[1]

            particle_x = np.clip(particle_x + dx, -4.9, 4.9)
            particle_y = np.clip(particle_y + dy, -4.9, 4.9)
            particle_theta += dtheta

            for i, particle in enumerate(self.particles):
                particle.x = particle_x[i]
                particle.y = particle_y[i]
                particle.theta = particle_theta[i]
                particle.scan[:] = 2.2
            # virtual scan & calc weight
            self.virtual_scan(scan_vec)
            self.adaptive_resample()           
        self.visualize(loc, scan)
        self.loc_prev = loc


    def icp_update(self, loc, scan):
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

            particle_x = np.array([p.x for p in self.particles])
            particle_y = np.array([p.y for p in self.particles])
            particle_theta = np.array([p.theta for p in self.particles])

            cos_p_theta = np.cos(particle_theta)
            sin_p_theta = np.sin(particle_theta)
            dx = cos_p_theta * dr[0] - sin_p_theta * dr[1]
            dy = sin_p_theta * dr[0] + cos_p_theta * dr[1]

            particle_x = np.clip(particle_x + dx, -4.9, 4.9)
            particle_y = np.clip(particle_y + dy, -4.9, 4.9)
            particle_theta += dtheta

            for i, particle in enumerate(self.particles):
                particle.x = particle_x[i]
                particle.y = particle_y[i]
                particle.theta = particle_theta[i]
                particle.scan[:] = 2.2
            # virtual scan
            self.virtual_scan(scan_vec)

            # iterative closet point
            for particle in self.particles:
                real_scan = np.vstack([np.cos(self.scan_theta) * scan_vec, np.sin(self.scan_theta) * scan_vec]).T
                virtual_scan = np.vstack([np.cos(self.scan_theta) * particle.scan, np.sin(self.scan_theta) * particle.scan]).T
                R, t = self.icp(real_scan, virtual_scan)

                position = np.array([particle.x, particle.y]).T
                new_position = np.dot(R, position) + t
                particle.x, particle.y = new_position
                particle.theta += np.arctan2(R[1,0], R[0,0])

            # calculate weights
            self.adaptive_resample()           
        self.visualize(loc, scan)
        self.loc_prev = loc




    def rk4_update(self, dt, v, omega, loc, scan):
        scan_vec = np.array(
            [data[1] if data[0] == 1 else 2.2 for data in scan]
        )

        if self.loc_prev:
            prev_theta = self.loc_prev[2]
            dr_x, dr_y = loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]
            dtheta = loc[2] - self.loc_prev[2]

            cos_theta = np.cos(-prev_theta)
            sin_theta = np.sin(-prev_theta)
            dr = np.array([cos_theta * dr_x - sin_theta * dr_y, 
                           sin_theta * dr_x + cos_theta * dr_y])

            particle_x = np.array([p.x for p in self.particles])
            particle_y = np.array([p.y for p in self.particles])
            particle_theta = np.array([p.theta for p in self.particles])

            # RK4
            dt = 0.1
            v = np.linalg.norm(dr) / dt
            omega = dtheta / dt
            
            for i in range(self.num_particles):
                x0, y0, theta0 = particle_x[i], particle_y[i], particle_theta[i]

                v_noise = v + np.random.randn() * 0.01
                omega_noise = omega + np.random.randn() * 0.01

                k1_x = v_noise * np.cos(theta0)
                k1_y = v_noise * np.sin(theta0)
                k1_theta = omega_noise

                k2_x = v_noise * np.cos(theta0 + 0.5 * dt * k1_theta)
                k2_y = v_noise * np.sin(theta0 + 0.5 * dt * k1_theta)
                k2_theta = omega_noise

                k3_x = v_noise * np.cos(theta0 + 0.5 * dt * k2_theta)
                k3_y = v_noise * np.sin(theta0 + 0.5 * dt * k2_theta)
                k3_theta = omega_noise

                k4_x = v_noise * np.cos(theta0 + dt * k3_theta)
                k4_y = v_noise * np.sin(theta0 + dt * k3_theta)
                k4_theta = omega_noise

                particle_x[i] += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                # particle_x[i] = np.clip(particle_x[i], -4.99, 4.99)
                particle_y[i] += (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
                # particle_y[i] = np.clip(particle_y[i], -4.99, 4.99)
                particle_theta[i] += (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

                particle_theta[i] = (particle_theta[i] + np.pi) % (2 * np.pi) - np.pi

            particle_x = np.clip(particle_x, -4.9, 4.9)
            particle_y = np.clip(particle_y, -4.9, 4.9)
            
            for i, particle in enumerate(self.particles):
                particle.x = particle_x[i]
                particle.y = particle_y[i]
                particle.theta = particle_theta[i]
                particle.scan[:] = 2.2

            # virtual scan & calc weight
            self.virtual_scan(scan_vec)
            self.adaptive_resample()           
        self.visualize(loc, scan)
        self.loc_prev = loc


    def virtual_scan(self, scan_vec):
       for particle in self.particles:
            # range
            dist = 2.25
            i_min = max(0, int((particle.x - dist) // 0.1 + 50))
            i_max = min(99, int((particle.x + dist) // 0.1 + 50))
            j_min = max(0, int((particle.y - dist) // 0.1 + 50))
            j_max = min(99, int((particle.y + dist) // 0.1 + 50))

            # sub grid
            sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]

            # x distance
            gx = np.arange(i_min, i_max + 1) * 0.1 + 0.05 - 5
            gx = np.repeat(gx.reshape(1, -1), sub_grid.shape[0], axis=0)
            dx = gx - particle.x

            # y distance
            gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
            gy = np.repeat(gy.reshape(1, -1).T, sub_grid.shape[1], axis=1)
            dy = gy - particle.y

            # distance
            gd = (dx**2 + dy**2) ** 0.5

            # theta diff
            gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 - 1)
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
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                area_grid = sub_grid[area]
                area_dist = gd[area]
                assert area_grid.shape == area_dist.shape
                area_valid = area_grid > 0
                if area_valid.shape[0] > 0 and np.max(area_valid) > 0:
                    particle.scan[i] = np.min(area_dist[area_valid])
            error = np.linalg.norm(scan_vec - particle.scan)
            particle.weight = 0.1 / (error + 1e-2)
            particle.weight = np.clip(particle.weight, 1e-3, 1.0)
            

    def adaptive_resample(self):
        # calculate weights
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)

        # resample based on weights
        particles = np.random.choice(self.particles, len(self.particles), p=weights)
        self.particles = [copy.deepcopy(p) for p in particles]

        # calculate the KLD-sampling dynamic number of particles
        eff_particles = 1 / np.sum(weights**2)
        if eff_particles <= 0 or np.isnan(eff_particles):
            eff_particles = 1       # prevent zero or NaN values

        # calculate KLD-sampling dynamic number of particles
        try:
            kld_error = np.sqrt(self.epsilon * (1-eff_particles) / eff_particles)
            if np.isnan(kld_error):
                kld_error = 0
            print(f'eff_particles = {eff_particles}')
            # print(f'kld_error : {kld_error}')
        except ValueError:
            kld_error = 0       # ensure kld_error is non-negative
        new_num_particles = min(self.max_particles, max(self.min_particles, int(self.num_particles * (1 + kld_error))))
        
        # adjust the number of particles
        if new_num_particles != self.num_particles:
            self.num_particles = new_num_particles
            self.particles = [Particle() for _ in range(self.num_particles)]

        # add noise for new particles
        for particle in self.particles:
            particle.x += np.random.randn() * self.sigma
            particle.y += np.random.randn() * self.sigma
            particle.theta = np.random.randn() * self.sigma
        self.sigma = max(self.sigma * 0.99, 0.015)


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
        x = [p.x for p in self.particles]
        y = [p.y for p in self.particles]
        c = [p.weight for p in self.particles]
        self.plt_objects[15] = plt.scatter(x, y, s=3**2, c=c, cmap="Spectral")

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = LocalizationBot()
    client.init_coppelia()
    client.run_coppelia()
