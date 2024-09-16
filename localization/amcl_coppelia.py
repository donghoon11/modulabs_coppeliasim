import copy
import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot


class LocalizationBot(YouBot):
    def __init__(self):
        super().__init__()
        self.amcl = AdvancedAdaptiveMonteCarloLocalization()

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
        # update grid
        self.amcl.update(loc, scan)


class Particle:
    def __init__(self):
        self.x: float = np.random.uniform(-5, 5)
        self.y: float = np.random.uniform(-5, 5)
        self.theta: float = np.random.uniform(-np.pi, np.pi)
        self.scan: np.array = np.full(13, 2.2)
        self.weight: float = 0.01

    def add_noise(self, move_noise, turn_noise):
        self.x += np.random.randn() * move_noise
        self.y += np.random.randn() * move_noise
        self.theta += np.random.randn() * turn_noise


class AdvancedAdaptiveMonteCarloLocalization:
    def __init__(self):
        # Particle settings
        self.particles = [Particle() for _ in range(200)]
        self.min_particles = 50
        self.max_particles = 2000
        self.move_noise = 0.05  # 노이즈 추가
        self.turn_noise = 0.02  # 회전 시 노이즈 추가
        self.threshold = 0.3  # Resampling threshold
        self.particle_expansion_rate = 50

        # Grid map
        with open("/home/oh/my_coppeliasim/modulabs_coppeliasim/mapping/mapping.npy", "rb") as f:
            self.grid = np.load(f)

        # Plotting grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)
        self.plt_objects = [None] * (15 + 1 + 13)  # For grid, robot, scans, particles

        # Scanner and delta configuration
        self.delta = np.pi / 12
        self.scan_theta = np.array([-np.pi / 2 + self.delta * i for i in range(13)])
        self.boundary = np.pi / 2 + self.delta / 2
        self.loc_prev = None

        # Parameters for adaptive resampling
        self.sigma = 1.0
        self.effective_sample_size_threshold = 0.5  # For adaptive resampling

    def update(self, loc, scan):
        scan_vec = np.array([data[1] if data[0] == 1 else 2.2 for _, data in enumerate(scan)])
        if self.loc_prev:
            prev_theta = self.loc_prev[2]
            dr = np.array([[np.cos(-prev_theta), -np.sin(-prev_theta)],
                           [np.sin(-prev_theta), np.cos(-prev_theta)]]
                          ).dot(np.array([loc[0] - self.loc_prev[0], loc[1] - self.loc_prev[1]]))
            dtheta = loc[2] - self.loc_prev[2]

            # Update particle positions with noise
            for particle in self.particles:
                dp = np.array([[np.cos(particle.theta), -np.sin(particle.theta)],
                               [np.sin(particle.theta), np.cos(particle.theta)]]
                              ).dot(dr)
                particle.x += dp[0]
                particle.y += dp[1]
                particle.theta += dtheta

                # Apply noise to movement and turning
                particle.add_noise(self.move_noise, self.turn_noise)

            # Simulate scan & update weights
            self.virtual_scan(scan_vec)
            self.resample()
            self.adjust_particle_count(scan_vec)

        self.visualize(loc, scan)
        self.loc_prev = loc

    def virtual_scan(self, scan_vec):
        for particle in self.particles:
            dist = 2.25
            i_min = max(0, int((particle.x - dist) // 0.1 + 50))
            i_max = min(99, int((particle.x + dist) // 0.1 + 50))
            j_min = max(0, int((particle.y - dist) // 0.1 + 50))
            j_max = min(99, int((particle.y + dist) // 0.1 + 50))

            sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]
            gx = np.arange(i_min, i_max + 1) * 0.1 + 0.05 - 5
            gx = np.repeat(gx.reshape(1, -1), sub_grid.shape[0], axis=0)
            dx = gx - particle.x

            gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
            gy = np.repeat(gy.reshape(1, -1).T, sub_grid.shape[1], axis=1)
            dy = gy - particle.y

            gd = (dx**2 + dy**2) ** 0.5
            gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 - 1)
            dtheta = gtheta - particle.theta

            while np.pi < np.max(dtheta):
                dtheta -= (np.pi < dtheta) * 2 * np.pi
            while np.min(dtheta) < -np.pi:
                dtheta += (dtheta < -np.pi) * 2 * np.pi

            for i in range(13):
                area = (
                    (gd < dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                area_grid = sub_grid[area]
                area_dist = gd[area]
                area_valid = area_grid > 0
                if area_valid.shape[0] > 0 and np.max(area_valid) > 0:
                    particle.scan[i] = np.min(area_dist[area_valid])

            particle.weight = 0.1 / (np.linalg.norm(scan_vec - particle.scan) + 1e-2)

    def resample(self):
        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)
        eff_N = 1.0 / np.sum(weights**2)  # Effective particle count

        if eff_N < self.effective_sample_size_threshold * len(self.particles):
            particles = np.random.choice(self.particles, len(self.particles), p=weights)
            particles = [copy.deepcopy(particle) for particle in particles]

            for particle in particles:
                particle.add_noise(self.move_noise, self.turn_noise)

            self.sigma = max(self.sigma * 0.99, 0.015)
            self.particles = particles

    def adjust_particle_count(self, scan_vec):
        weights = np.array([particle.weight for particle in self.particles])
        variance = np.var(weights)

        if variance < self.threshold and len(self.particles) < self.max_particles:
            new_particles = [Particle() for _ in range(self.particle_expansion_rate)]
            self.particles.extend(new_particles)
        elif variance > self.threshold and len(self.particles) > self.min_particles:
            self.particles = self.particles[:len(self.particles) - self.particle_expansion_rate]

    def visualize(self, loc, scan):
        x, y, theta = loc
        for object in self.plt_objects:
            if object:
                object.remove()

        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")

        (self.plt_objects[1],) = plt.plot(x, y, color="green", marker="o", markersize=10)

        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data
            res = res > 0
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

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
