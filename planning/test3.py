import math
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class MobileRobotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        self.frontRefHandle = self.sim.getObject('/youBot_frontRef')
        self.motor_fl = self.sim.getObject('/rollingJoint_fl')
        self.motor_rl = self.sim.getObject('/rollingJoint_rl')
        self.motor_fr = self.sim.getObject('/rollingJoint_fr')
        self.motor_rr = self.sim.getObject('/rollingJoint_rr')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        self.goalDummyHandle = self.sim.getObject('/youBot_goalDummy')

        self.velocity = 180 * math.pi / 180
        self.searchRange = 5
        self.searchDuration = 0.1
        self.searchAlgo = self.simOMPL.Algorithm.BiTRRT
        self.displayCollisionFreeNodes = True
        self.showRealTarget = True
        self.showTrackPos = True
        self.line_container = None
        self.pt_cont = None

        self.robotObstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        symbol = True if collision == 0 else False
        return symbol

    def get_target_position(self):
        return self.sim.getObjectPosition(self.goalDummyHandle, -1)

    def visualize_collision_free_nodes(self, states):
        if self.pt_cont is None:
            self.pt_cont = self.sim.addDrawingObject(self.sim.drawing_spherepoints, 0.05, 0, -1, 0, [0, 1, 0])
        else:
            self.sim.addDrawingObjectItem(self.pt_cont, None)

        if states:
            print(states)
            print(f'len(states) : {len(states)}')
            for i in range(0, len(states)+1):
                self.sim.addDrawingObjectItem(self.pt_cont, [states[i], states[i+1], 0.025])

    def visualize_path(self, path):
        if self.line_container is None:
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])
        self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path) // 2):
                line_data = [path[2*i], path[2*i+1], 0.001, path[2*i-2], path[2*i-1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

    def plan_path(self, goal_position):
        path = None

        task = self.simOMPL.createTask('task')
        self.simOMPL.setAlgorithm(task, self.searchAlgo)
        start_pos = self.sim.getObjectPosition(self.refHandle, -1)
        ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                            [start_pos[0] - self.searchRange, start_pos[1] - self.searchRange],
                                            [start_pos[0] + self.searchRange, start_pos[1] + self.searchRange], 1)]
        self.simOMPL.setStateSpace(task, ss)
        self.simOMPL.setCollisionPairs(task, self.collPairs)
        self.simOMPL.setStartState(task, start_pos[:2])
        self.simOMPL.setGoalState(task, goal_position[:2])
        self.simOMPL.setStateValidityCheckingResolution(task, 0.001)
        self.simOMPL.setup(task)

        while path is None:
            if self.simOMPL.solve(task, self.searchDuration):
                self.simOMPL.simplifyPath(task, self.searchDuration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)
                if self.displayCollisionFreeNodes:
                    states = self.simOMPL.getPlannerData(task)
                    self.visualize_collision_free_nodes(states)

            goal_moved = np.linalg.norm(np.array(self.get_target_position()) - np.array(goal_position)) > 0.1
            if goal_moved:
                break

        return path

    def smooth_velocity(self, target, current, alpha):
        return current + alpha * (target - current)
    
    def omni_wheel_control(self, v_forward, v_side, v_turn):
        """
        Control the mecanum wheels, supporting lateral movement.
        :param v_forward: Forward/Backward velocity
        :param v_turn: Rotational velocity
        :param v_side: Lateral velocity (for side movement)
        """
        # params for 4 mecanum wheel drive
        radius = 0.05       # wheel radius
        dist_R = 0.228 + 0.158      # (distance b.w. centroid & wheel cent.) = dist_x + dist_y

        # Calculate wheel velocities for mecanum drive
        v_forward = - v_forward

        fl_speed = (v_forward - v_side - v_turn * dist_R ) / radius
        rl_speed = (v_forward + v_side - v_turn * dist_R ) / radius
        fr_wheel_speed = (v_forward + v_side + v_turn * dist_R) / radius
        rr_wheel_speed = (v_forward - v_side + v_turn * dist_R ) / radius

        # Set motor velocities
        self.sim.setJointTargetVelocity(self.motor_fl, fl_speed)
        self.sim.setJointTargetVelocity(self.motor_rl, rl_speed)
        self.sim.setJointTargetVelocity(self.motor_fr, fr_wheel_speed)
        self.sim.setJointTargetVelocity(self.motor_rr, rr_wheel_speed)


    def follow_path(self, path):
        path_3d = []
        for i in range(0, len(path), 2):
            path_3d.append([path[i], path[i+1], 0.0])

        track_pos_cont = None
        if self.showTrackPos:
            track_pos_cont = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])

        prev_l = 0
        while True:
            path_lengths, total_dist = self.sim.getPathLengths(path_3d, 3)
            closest_l = self.sim.getClosestPosOnPath(path_3d, path_lengths, self.sim.getObjectPosition(self.frontRefHandle, -1))
            if closest_l <= prev_l:
                closest_l += total_dist / 200
            prev_l = closest_l

            target_point = self.sim.getPathInterpolatedConfig(path_3d, path_lengths, closest_l)
            if track_pos_cont:
                self.sim.addDrawingObjectItem(track_pos_cont, target_point)

            # Get relative position of target point
            m = self.sim.getObjectMatrix(self.refHandle, -1)
            self.sim.getMatrixInverse(m)
            relative_target = self.sim.multiplyVector(m, target_point)

            target_x, target_y = relative_target[0], relative_target[1]

            # Compute angle for rotation

            pos = self.sim.getObjectPosition(self.refHandle, -1)

            v_x = 0.5 * (target_x - pos[0])
            v_y = 0.5 * (target_y - pos[1])
            angle = math.atan2(relative_target[1], relative_target[0])

            self.omni_wheel_control(v_x, v_y, angle)

            # Stop the loop if the robot reaches the target position and orientation
            if np.linalg.norm([target_x - pos[0], target_y - pos[1]]) < 0.1 and abs(target_theta - orientation[2]) < 0.1:
                break

            time.sleep(0.01)

    def move_robot_to_target(self):
        # current_left_v = 0.0
        # current_right_v = 0.0

        while True:
            goal_position = self.get_target_position()
            
            # Adjust goal if collision is detected
            while self.check_collides_at(self.sim.getObjectPosition(self.refHandle, -1)):
                start_pos = np.array(self.sim.getObjectPosition(self.refHandle, -1))
                goal_pos = np.array(goal_position)

                direction = (goal_pos - start_pos) / np.linalg.norm(goal_pos - start_pos)
                distance = np.linalg.norm(goal_pos - start_pos)

                if distance > 0.1:
                    start_pos += direction * self.sim.getSimulationTimeStep() * self.velocity
                    self.sim.setObjectPosition(self.refHandle, -1, start_pos.tolist())
                    self.sim.setObjectOrientation(self.refHandle, -1, [0, 0, math.atan2(direction[1], direction[0])])

                time.sleep(0.01)
            
            # Motion planning
            path = self.plan_path(goal_position)
            if path:
                self.follow_path(path)
            
            time.sleep(0.01)

    def run_step(self):
        # self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            # Plan and follow the path
            self.move_robot_to_target()

            time.sleep(0.01)
            # self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    controller = MobileRobotPP()
    controller.init_coppelia()
    controller.run_step()