'''
fastslam 내용을 pioneer-hokuyo 에 적용.
rospy -> remoteapi 로 변경하여 사용.
'''


import math
import random
import numpy as np
from collections import deque
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class VrepEnvironment():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

    def init_coppelia(self):
        self.robot = self.sim.getObject('/PioneerP3DX')
        x, y = self.sim.getObjectPosition(self.robot)
        theta = self.sim.getObjectOrientation(self.robot)[2]

        self.visionSensors = []