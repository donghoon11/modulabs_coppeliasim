import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Landmarks
landmarks = np.array([[20.0, 20.0], [20.0, 80.0], [20.0, 50.0],
                      [50.0, 20.0], [50.0, 80.0], [80.0, 80.0],
                      [80.0, 20.0], [80.0, 50.0]])

# Map size in meters
world_size = 100.0

# Global Functions
def mod(first_term, second_term):
    return first_term - (second_term * math.floor(first_term / second_term))

def gen_real_random():
    return random.uniform(0.0, 1.0)

class Robot:
    def __init__(self):
        self.x = gen_real_random() * world_size
        self.y = gen_real_random() * world_size
        self.orient = gen_real_random() * 2.0 * math.pi
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self, new_x, new_y, new_orient):
        if new_x < 0 or new_x >= world_size:
            raise ValueError("X coordinate out of bounds")
        if new_y < 0 or new_y >= world_size:
            raise ValueError("Y coordinate out of bounds")
        if new_orient < 0 or new_orient >= 2 * math.pi:
            raise ValueError("Orientation must be in [0..2pi]")

        self.x = new_x
        self.y = new_y
        self.orient = new_orient

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        self.forward_noise = new_forward_noise
        self.turn_noise = new_turn_noise
        self.sense_noise = new_sense_noise

    def sense(self):
        z = []
        for landmark in landmarks:
            dist = math.sqrt((self.x - landmark[0])**2 + (self.y - landmark[1])**2)
            dist += random.gauss(0.0, self.sense_noise)
            z.append(dist)
        return z

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError("Robot cannot move backward")

        orient = self.orient + turn + random.gauss(0.0, self.turn_noise)
        orient = mod(orient, 2 * math.pi)

        dist = forward + random.gauss(0.0, self.forward_noise)
        x = self.x + (math.cos(orient) * dist)
        y = self.y + (math.sin(orient) * dist)

        x = mod(x, world_size)
        y = mod(y, world_size)

        res = Robot()
        res.set(x, y, orient)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

        return res

    def show_pose(self):
        return f"[x={self.x} y={self.y} orient={self.orient}]"

    def read_sensors(self):
        z = self.sense()
        readings = "[" + " ".join(map(str, z)) + "]"
        return readings

    def measurement_prob(self, measurement):
        prob = 1.0
        for i in range(len(landmarks)):
            dist = math.sqrt((self.x - landmarks[i][0])**2 + (self.y - landmarks[i][1])**2)
            prob *= self.gaussian(dist, self.sense_noise, measurement[i])
        return prob

    def gaussian(self, mu, sigma, x):
        return math.exp(-((mu - x)**2) / (2 * (sigma**2))) / math.sqrt(2.0 * math.pi * sigma**2)

def evaluation(r, p, n):
    sum_err = 0.0
    for i in range(n):
        dx = mod(p[i].x - r.x + (world_size / 2.0), world_size) - (world_size / 2.0)
        dy = mod(p[i].y - r.y + (world_size / 2.0), world_size) - (world_size / 2.0)
        err = math.sqrt(dx**2 + dy**2)
        sum_err += err
    return sum_err / n

def max_array(arr):
    return max(arr)

def visualization(n, robot, step, p, pr):
    save_order = [0, 1, 2, 5, 10, 20, 40, 49]
    if (step in save_order):
        plt.title(f"MCL, step {step}")
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        for i in range(n):
            plt.plot(p[i].x, p[i].y, "go")

        for i in range(n):
            plt.plot(pr[i].x, pr[i].y, "yo")

        for landmark in landmarks:
            plt.plot(landmark[0], landmark[1], "ro")

        plt.plot(robot.x, robot.y, "bo")
        
        plt.savefig(f"./Images/Step{step}.png")
        plt.clf()
    else:
        pass

def main():
    myrobot = Robot()
    myrobot.set_noise(5.0, 0.1, 5.0)
    myrobot.set(30.0, 50.0, math.pi / 2.0)
    myrobot.move(-math.pi / 2.0, 15.0)
    myrobot.move(-math.pi / 2.0, 10.0)

    n = 1000    # 초기 파티클 입자 1000개 생성.
    particles = [Robot() for _ in range(n)]
    for i in range(n):
        particles[i].set_noise(0.05, 0.05, 5.0)

    myrobot = Robot()
    steps = 50

    for t in range(steps):
        myrobot = myrobot.move(0.1, 5.0)
        z = myrobot.sense()

        p2 = [particles[i].move(0.1, 5.0) for i in range(n)]

        w = [particles[i].measurement_prob(z) for i in range(n)]

        p3 = []
        index = int(gen_real_random() * n)
        beta = 0.0
        mw = max_array(w)

        for i in range(n):
            beta += gen_real_random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = int(mod(index + 1, n))
            p3.append(particles[index])

        particles = p3

        print(f"Step = {t}, Evaluation = {evaluation(myrobot, particles, n)}")

        visualization(n, myrobot, t, p2, p3)

if __name__ == "__main__":
    main()
