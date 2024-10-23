from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time
from scipy.optimize import minimize

class MobileRobotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        self.frontRefHandle = self.sim.getObject('/youBot_frontRef')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')  # Collision volume
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

        self.motor_fl = self.sim.getObject('/rollingJoint_fl')
        self.motor_rl = self.sim.getObject('/rollingJoint_rl')
        self.motor_fr = self.sim.getObject('/rollingJoint_fr')
        self.motor_rr = self.sim.getObject('/rollingJoint_rr')

        self.robotObstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 180 * math.pi / 180
        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1
        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

        # MPC params
        self.horizon = 10
        self.dt = 0.1

    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        print(collision)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        symbol = True if collision == 0 else False
        return symbol

    def get_target_position(self):
        """Returns the position of the goal dummy object."""
        return self.sim.getObjectPosition(self.goalDummyHandle, -1)

    def visualize_path(self, path):
        if self.line_container is None:
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])
        self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path) // 2):
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

    def move_robot_to_position(self, target_position):
        path = None
        
        while not path:
            task = self.simOMPL.createTask('t')
            self.simOMPL.setAlgorithm(task, self.search_algo)

            start_pos = self.sim.getObjectPosition(self.refHandle, -1)

            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0] - self.search_range, start_pos[1] - self.search_range],
                                                [start_pos[0] + self.search_range, start_pos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
            self.simOMPL.setup(task)

            # 경로 단순화 (! 여러 path 중 가장 단순한 path = 최적화 경로)
            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)

            time.sleep(0.01)
        return path

    def robot_kinematics(self, state, control):
        """
        Kinematic model for the mecanum wheel robot.
        :param state: [x, y, theta]
        :param control: [v_forward, v_side, v_turn]
        :return: next state [x_next, y_next, theta_next]
        """
        x, y, theta = state[0], state[1], state[3]
        v_forward, v_side, v_turn = control

        # Update the state based on kinematics equations
        x_next = x + v_forward * math.cos(theta) * self.dt - v_side * math.sin(theta) * self.dt
        y_next = y + v_forward * math.sin(theta) * self.dt + v_side * math.cos(theta) * self.dt
        theta_next = theta + v_turn * self.dt

        return [x_next, y_next, theta_next]

    def objective_function(self, controls, *args):
        """
        Objective function for the MPC optimization.
        :param controls: Flattened control inputs over the horizon
        :param args: Additional arguments: current state, reference path
        :return: Cost (to minimize)
        """
        state, ref_path = args
        cost = 0
        controls = np.reshape(controls, (self.horizon, 3))  # [v_forward, v_side, v_turn] per step
        print(controls)

        # Simulate the future states over the prediction horizon
        for i in range(self.horizon):
            control = controls[i]
            state = self.robot_kinematics(state, control)
            ref_state = ref_path   # type 이 바뀌어 버림.
            # Cost is the squared error between the predicted and reference positions
            cost += (state[0] - ref_state[0])**2 + (state[1] - ref_state[1])**2

        return cost

    def mpc_control(self, current_state, ref_path):
        """
        MPC control method to compute optimal control inputs.
        :param current_state: [x, y, theta] Current state of the robot
        :param ref_path: List of reference states over the horizon
        :return: Optimal control input [v_forward, v_side, v_turn]
        """
        # Initial guess for the controls [v_forward, v_side, v_turn] for each time step
        initial_controls = np.zeros((self.horizon, 3)).flatten()

        # Define bounds for the controls (can be tuned based on robot limits)
        bounds = [(-1, 1), (-1, 1), (-np.pi, np.pi)] * self.horizon

        # Optimize the control inputs using an optimization solver (minimize)
        result = minimize(self.objective_function, initial_controls, args=(current_state, ref_path),
                          bounds=bounds, method='SLSQP')

        # Extract the first control input from the optimized result
        optimal_controls = np.reshape(result.x, (self.horizon, 3))
        return optimal_controls[0]

    def omni_wheel_control(self, v_forward, v_side, v_turn):
        # params for 4 mecanum wheel drive
        radius = 0.05       # wheel radius
        dist_R = 0.228 + 0.158      # (distance b.w. centroid & wheel cent.) = dist_x + dist_y

        # Calculate wheel velocities for mecanum drive
        fl_speed = (-v_forward - v_side - v_turn * dist_R ) / radius
        rl_speed = (-v_forward + v_side - v_turn * dist_R ) / radius
        fr_wheel_speed = (-v_forward + v_side + v_turn * dist_R) / radius
        rr_wheel_speed = (-v_forward - v_side + v_turn * dist_R ) / radius

        # Set motor velocities
        self.sim.setJointTargetVelocity(self.motor_fl, fl_speed)
        self.sim.setJointTargetVelocity(self.motor_rl, rl_speed)
        self.sim.setJointTargetVelocity(self.motor_fr, fr_wheel_speed)
        self.sim.setJointTargetVelocity(self.motor_rr, rr_wheel_speed)

    def follow_path(self, path):
        """
        Follow the path using MPC control.
        :param path: Path as a list of (x, y, theta) coordinates
        """
        current_state = self.sim.getObjectPosition(self.refHandle, -1)
        current_state.append(self.sim.getObjectOrientation(self.refHandle, -1)[2])  # Add theta

        while True:
            # Define the reference path for the horizon
            ref_path = self.get_reference_path(path, current_state)

            # Get optimal control input from MPC
            v_forward, v_side, v_turn = self.mpc_control(current_state, ref_path)

            # Apply the control to the mecanum wheels
            self.omni_wheel_control(v_forward, v_side, v_turn)

            # Update current state for the next iteration
            current_state = self.robot_kinematics(current_state, [v_forward, v_side, v_turn])

            # Stop if the robot reaches the goal
            if np.linalg.norm(np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1)) -
                              np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.05:
                break

            time.sleep(0.01)

    def get_reference_path(self, path, current_state):
        """
        Get the reference path for the next horizon steps.
        :param path: Full path as (x, y, theta) coordinates
        :param current_state: Current state of the robot
        :return: Reference path for the horizon
        """
        ref_path = []
        for i in range(self.horizon):
            # Choose future path points (can be spaced out based on the lookahead distance)
            index = min(i, len(path) - 1)
            ref_path.append(path[index])
        return ref_path

    def run_step(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            goal_position = self.get_target_position()

            # Adjust goal if collision is detected
            while self.check_collides_at(goal_position):
                goal_position[0] -= 0.09

            # Plan and follow the path
            print('1')
            path = self.move_robot_to_position(goal_position)
            if path:
                self.follow_path(path)

            # Stop movement
            self.omni_wheel_control(0, 0, 0)

            time.sleep(0.01)
            self.sim.step()
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = MobileRobotPP()
    controller.init_coppelia()
    controller.run_step()
