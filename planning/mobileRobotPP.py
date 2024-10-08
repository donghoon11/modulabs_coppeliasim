'''
### Key Functions

- **Collision Detection** (`checkCollidesAt`):
    - Checks if the robot's collision volume is in a colliding state at a given position.
- **Path Visualization** (`visualizePath` and `visualizeCollisionFreeNodes`):
    - Draws the computed path and any collision-free nodes on the simulation interface for debugging and visualization.
- **Target Position Handling** (`getTargetPosition`):
    - Retrieves the current target position of the robot, incorporating some latency handling for stability.

### Main Control Logic (`coroutineMain`)

1. **Collision Handling**:
    - Ensures that the robot starts from a non-colliding position. If it does, it attempts to adjust its position until it is safe.
2. **Goal Management**:
    - Determines a goal position and checks its distance from the robot. If the goal is too far or colliding, it adjusts the goal position.
3. **Path Planning**:
    - If the goal is reachable, it creates a path using the OMPL (Open Motion Planning Library) and visualizes it.
    - It handles potential goal movements during planning, re-computing paths if necessary.
4. **Movement Execution**:
    - Once a path is found, the robot tracks the path by actuating its motors toward the target points.
    - Adjusts the velocities of the left and right wheels based on the desired direction.
5. **Stopping Conditions**:
    - Stops the robot when it reaches the goal or if the goal moves significantly.
'''
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np

class MobileRobotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject(self.sim.handle_self)
        self.refHandle = self.sim.getObject('dr12_ref')
        self.frontRefHandle = self.sim.getObject('dr12_frontRef')
        self.leftMotorHandle = self.sim.getObject('dr12_leftJoint')
        self.rightMotorHandle = self.sim.getObject('dr12_rightJoint')
        self.collVolumeHandle = self.sim.getObject('dr12_coll')     # dr12 의 boundary box
        self.goalDummyHandle = self.sim.getObject('dr12_goalDummy')

        self.robotObstaclesCollection = self.sim.createCollection(0)        # 충돌 검사용 객체 컬렉션
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.velocity = 180 * math.pi / 180     # 바퀴 회전 속도
        self.searchRange = 5
        self.searchAlgo = self.simOMPL.Algorithm.BiTRRT
        self.displayCollisionFreeNodes = True
        self.showRealTarget = True

        # 
        corout = coroutine.create(self.coroutine_main())


    def check_collides_at(self, pos):
        tmp = self.sim.getObjectPosition(self.collVolumeHandle, -1)
        self.sim.setObjectPosition(self.collVolumeHandle, -1, pos)
        r = self.sim.checkCollision(self.collPairs[0], self.collPairs[1])
        self.sim.setObjectPosition(self.collVolumeHandle, -1, tmp)
        return r > 0


    def visualize_collision_free_nodes(self, states):
        if ptCont:
            self.sim.addDrawingObjectItem(ptCont, None)
        else:
            ptCont = self.sim.addDrawingObject(self.sim.drawing_spherepoints, 0.05, 0, -1, 0, [0, 1, 0])
        if states:
            for i in range(len(states) // 2):
                self.sim.addDrawingObjectItem(ptCont, [states[2 * i], states[2 * i + 1], 0.025])


    def get_target_position(self):
        p = np.array(self.sim.getObjectPosition(self.goalDummyHandle, -1))
        t = self.sim.getSystemTimeInMs(-1)

        if self.prevTargetPosStable is None:
            self.prevTargetPosStable = p
            self.prevTargetPos = p
            self.prevTimeDiff = t

        if np.linalg.norm(self.prevTargetPos - p) > 0.01:
            self.prevTimeDiff = t

        self.prevTargetPos = p
        if self.sim.getSystemTimeInMs(self.prevTimeDiff) > 250:
            self.prevTargetPosStable = p

        return self.prevTargetPosStable


    def visualize_path(self, path):
        if not _lineContainer:
            _lineContainer = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])
        self.sim.addDrawingObjectItem(_lineContainer, None)
        if path:
            for i in range(len(path) // 2 - 1):
                lineDat = [path[i * 2], path[i * 2 + 1], 0.001, path[(i + 1) * 2], path[(i + 1) * 2 + 1], 0.001]
                self.sim.ddDrawingObjectItem(_lineContainer, lineDat)


    def coroutine_main(self):
        # global velocity, searchRange, searchDuration, searchAlgo, collPairs, refHandle, goalDummyHandle, collVolumeHandle, frontRefHandle
        self.sim.setThreadAutomaticSwitch(False)
        i = 0

        while True:
            i += 1
            while self.check_collides_at(self.sim.getObjectPosition(self.refHandle, -1)):
                sp = np.array(self.sim.getObjectPosition(self.refHandle, -1))
                gp = np.array(self.get_target_position())
                dx = (gp - sp) / np.linalg.norm(gp - sp)
                l = np.linalg.norm(gp - sp)

                if l > 0.1:
                    sp = sp + dx * self.sim.getSimulationTimeStep() * self.velocity
                    self.sim.setObjectPosition(self.BillHandle, -1, sp)
                    self.sim.setObjectOrientation(self.BillHandle, -1, [0, 0, math.atan2(dx[1], dx[0])])
                self.sim.switchThread()

            sp = np.array(self.sim.getObjectPosition(self.refHandle, -1))
            gp = np.array(self.get_target_position())
            ogp = np.array(self.get_target_position())
            l = np.linalg.norm(gp - sp)
            ngo = False

            while l > 0.1 and (l > self.searchRange or self.check_collides_at(gp)):
                dx = (sp - gp) / np.linalg.norm(sp - gp)
                l = np.linalg.norm(gp - sp)
                gp = gp + dx * 0.09
                if self.showRealTarget:
                    ngo = True

            if ngo:
                ngo = self.sim.copyPasteObjects([self.goalDummyHandle], 1)[0]
                s = self.sim.getObjectsInTree(ngo, self.sim.object_shape_type)
                for obj in s:
                    self.sim.setShapeColor(obj, None, self.sim.colorcomponent_ambient_diffuse, [1, 0, 0])
                self.sim.setObjectPosition(ngo, -1, gp)

            if l > 0.1 and not self.check_collides_at(gp):
                t = self.simOMPL.createTask('t')
                self.simOMPL.setAlgorithm(t, self.searchAlgo)
                ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                            [sp[0] - self.searchRange, sp[1] - self.searchRange],
                                            [sp[0] + self.searchRange, sp[1] + self.searchRange], 1)]
                self.simOMPL.setStateSpace(t, ss)
                self.simOMPL.setCollisionPairs(t, self.collPairs)
                self.simOMPL.setStartState(t, [sp[0], sp[1]])
                self.simOMPL.setGoalState(t, [gp[0], gp[1]])
                self.simOMPL.setStateValidityCheckingResolution(t, 0.001)
                self.simOMPL.setup(t)

                path = None
                while path is None:
                    if self.simOMPL.solve(t, searchDuration):
                        self.sim.switchThread()
                        self.simOMPL.simplifyPath(t, searchDuration)
                        self.sim.switchThread()
                        path = self.simOMPL.getPath(t)
                        self.visualize_path(path)

                    if self.displayCollisionFreeNodes:
                        states = self.simOMPL.getPlannerData(t)
                        self.visualize_collision_free_nodes(states)
                    self.sim.switchThread()

                    gp2 = np.array(self.get_target_position())
                    l = np.linalg.norm(gp2 - ogp)
                    if l > 0.1:
                        break

                if path:
                    path = [val for sublist in path for val in sublist] + [0.0]
                    prevL = 0

                    while True:
                        pathLengths, totalDist = self.sim.getPathLengths(path, 3)
                        l = self.sim.getClosestPosOnPath(path, pathLengths, self.sim.getObjectPosition(self.frontRefHandle, -1))
                        if l <= prevL:
                            l += totalDist / 200
                        prevL = l

                        p = self.sim.getPathInterpolatedConfig(path, pathLengths, l)
                        if trackPosCont:
                            self.sim.addDrawingObjectItem(trackPosCont, p)

                        m = self.sim.getObjectMatrix(self.refHandle, -1)
                        self.sim.invertMatrix(m)
                        p = self.sim.multiplyVector(m, p)

                        angle = math.atan2(p[1], p[0])

                        leftV, rightV = 1.0, 1.0
                        if angle > 0.0:
                            leftV = max(1 - 4 * angle / math.pi, -1)
                        else:
                            rightV = max(1 + 4 * angle / math.pi, -1)

                        self.sim.setJointTargetVelocity(self.leftMotorHandle, leftV * self.velocity)
                        self.sim.setJointTargetVelocity(self.rightMotorHandle, rightV * self.velocity)
                        self.sim.switchThread()

                        gp2 = np.array(self.get_target_position())
                        l = np.linalg.norm(gp2 - ogp)
                        if l > 0.1:
                            break

                        pp = np.array(self.sim.getObjectPosition(self.refHandle, -1))
                        l = np.linalg.norm(gp2 - pp)
                        if l < 0.05:
                            break

                    if trackPosCont:
                        self.sim.removeDrawingObject(trackPosCont)

            if ngo:
                self.sim.removeModel(ngo)
            self.sim.switchThread()

    def sysCall_actuation():
        if not corout.done():
            corout.run()


