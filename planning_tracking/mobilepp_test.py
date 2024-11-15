from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

class MobileRobotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/dr12')
        self.refHandle = self.sim.getObject('/dr12_ref')
        self.frontRefHandle = self.sim.getObject('/dr12_frontRef')
        self.leftMotorHandle = self.sim.getObject('/dr12_leftJoint')
        self.rightMotorHandle = self.sim.getObject('/dr12_rightJoint')
        self.collVolumeHandle = self.sim.getObject('/dr12_coll')     
        self.goalDummyHandle = self.sim.getObject('/dr12_goalDummy1')
        self.goalDummyHandle2 = self.sim.getObject('/dr12_goalDummy2') 

        self.start_goal_handle = None       # added

        self.robotObstaclesCollection = self.sim.createCollection(0)       
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 0.2 * math.pi   # Changed the vel. param. for stable driving (slower than previous vel.; math.pi)
        self.search_range = 5
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1
        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

    # Collision checking is already reflected when path planning is executed.
    # The function is somewhat ambiguous, so I just didn't use it.
    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        collision = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        return collision


    def get_target_position(self, currentGoalHandle): # get target position for the current goal
        """Returns the position of the goal dummy object."""
        return self.sim.getObjectPosition(currentGoalHandle, -1)
    
    
    def visualize_path(self, path):
        """Visualizes the robot's path."""
        if self.line_container is None:     # if not self.line_container 
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])

        self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path)//2):
                line_data = [path[2*i-2], path[2*i-1], 0.001, path[2*i], path[2*i+1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)


    def move_robot_to_position(self, target_position):
        path = None
        while not path:
            task = self.simOMPL.createTask('t') 
            self.simOMPL.setAlgorithm(task, self.search_algo)

            start_pos = self.sim.getObjectPosition(self.refHandle, -1)
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [start_pos[0]-self.search_range, start_pos[1]-self.search_range],
                                                [start_pos[0]+self.search_range, start_pos[1]+self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, start_pos[:2])
            self.simOMPL.setGoalState(task, target_position[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.001) 
            self.simOMPL.setup(task)
            print('Try to solve')       # Add print statement for debugging
            if self.simOMPL.solve(task, self.search_duration):
                self.simOMPL.simplifyPath(task, self.search_duration)
                path = self.simOMPL.getPath(task)
                self.visualize_path(path)
            print('Done')               # Add print statement for debugging
            time.sleep(0.01)
        return path
    
    def follow_path(self, goal, path=None):
        if path is not None:
            path_3d = []
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])

            prev_l = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                current_pos = self.sim.getObjectPosition(self.frontRefHandle, -1)
                
                # Derive the closest point from the current location of the robot
                path_lengths, total_dist = self.sim.getPathLengths(path_3d, 3)
                closet_l = self.sim.getClosestPosOnPath(path_3d, path_lengths, current_pos)
                
                if closet_l <= prev_l:
                    closet_l += total_dist / 200
                prev_l = closet_l

                target_point = self.sim.getPathInterpolatedConfig(path_3d, path_lengths, closet_l)
                self.sim.addDrawingObjectItem(track_pos_container, target_point)

                # Relative position of the target position
                m = self.sim.getObjectMatrix(self.refHandle, -1)
                # Define the "inverse matrix"
                m_inv = self.sim.getMatrixInverse(m)
                relative_target = self.sim.multiplyVector(m_inv, target_point)

                # Compute angle (Z angle diff.))
                angle = math.atan2(relative_target[1], relative_target[0])

                # Adjust wheel velocities
                # 안정적인 주행을 위해 1.0, 4 와 같은 값들에 대해 파라미터 튜닝을 해야 할 것 같습니다.
                left_v = 1.0 - 4 * angle / math.pi if angle > 0 else 1.0
                right_v = 1.0 + 4 * angle / math.pi if angle <= 0 else 1.0
                left_v = max(-1.0, left_v)
                right_v = max(-1.0, right_v)

                self.sim.setJointTargetVelocity(self.leftMotorHandle, left_v * self.velocity)
                self.sim.setJointTargetVelocity(self.rightMotorHandle, right_v * self.velocity)

                # Break when close to the target
                if np.linalg.norm(np.array(self.sim.getObjectPosition(goal, -1)) - 
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.1:      # 0.05 >> 0.1
                    break

                time.sleep(1)       # set the long time sleep

    # Goal inputs from the user 
    def get_user_goal_choice(self):
        """Prompts the user to choose the goal."""   
        # start goal     
        while True:
            start_goal_input = int(input("Choose the start goal: (1) Goal 1 or (2) Goal 2: "))      # Define the int type
            if start_goal_input == 1:
                print("Start Goal: Goal 1 selected.")
                self.start_goal_handle = self.goalDummyHandle
                break

            elif start_goal_input == 2:
                print("Start Goal: Goal 2 selected.")
                self.start_goal_handle = self.goalDummyHandle2 
                break

            else:
                print("Invalid input. Please enter 1 or 2.")
        
        # end goal
        while True:
            end_goal_input = int(input("Choose the end goal: (1) Goal 1 or (2) Goal 2: "))
            try:
                if end_goal_input == 1 and self.start_goal_handle != 1:     # double check not to select the same place 
                    print("End Goal: Goal 1 selected.")
                    self.end_goal_handle = self.goalDummyHandle 
                    break
            except:
                print('Goal 1 is already selected.')
            try:
                if end_goal_input == 2 and self.start_goal_handle != 2:
                    print("End Goal: Goal 2 selected.")
                    self.end_goal_handle = self.goalDummyHandle2 
                    break
            except:
                print('Goal 2 is already selected')
            if end_goal_input != 1 and end_goal_input != 2:
                print("Invalid input. Please enter 1 or 2.")
        


    def run_step(self):
        self.sim.startSimulation(True)
        # Get the user's choice for start and end goals
        self.get_user_goal_choice()         # no return, just set the variables; self.start_goal_handle, self.end_goal_handle

        # Move to the start goal first
        goal_position = self.get_target_position(self.start_goal_handle)
        path = self.move_robot_to_position(goal_position)
        if path:
            self.follow_path(goal=self.start_goal_handle, path=path)
            print('check')            # Add the print statement for debugging

        # Remove the line and init the statespace for the another path planning
        self.sim.removeDrawingObject(self.line_container)
        self.line_container = None

        # fter reaching the start goal, move to the end goal
        goal_position = self.get_target_position(self.end_goal_handle)
        path = self.move_robot_to_position(goal_position)
        if path:
            self.follow_path(goal=self.end_goal_handle, path=path)
            print('check')            # Add the print statement for debugging

        # Remove the line and init the statespace for the another path planning
        self.sim.removeDrawingObject(self.line_container)
        self.line_container = None

        self.sim.stopSimulation()


if __name__ == "__main__":
    controller = MobileRobotPP()
    controller.init_coppelia()
    controller.run_step()
