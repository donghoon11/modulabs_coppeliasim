import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class HokuyoLidar:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

    def init_coppelia(self):
        self.maxScanDistance = 5
        self.showLines = True
        self.generateData = True
        self.rangeData = True  # if False, then X/Y/Z data rel to sensor base
        self.discardMaxDistPts = True
        
        # Get sensor and object handles
        self.self_handle = self.sim.getObject("/PioneerP3DX")
        self.visionSensors = [
            self.sim.getObject("/sensor1"),
            self.sim.getObject("/sensor2"),
        ]
        
        # Create a collection
        self.collection = self.sim.createCollection(0)  # 객체 컬렉션 생성. 여러 객체를 그룹으로 묶어 처리.
        '''
        sim.addItemToCollection(int collectionHandle, int what, int objectHandle, int options)
        
        * what : sim.handle_single, sim.handle_all, sim.handle_tree, sim.handle_chain 
        '''
        self.sim.addItemToCollection(self.collection, self.sim.handle_all, -1, 0)       # 시뮬레이션 내 모든 객체를 컬렉션에 추가.
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.self_handle, 1)    # 로봇과 그 하위 객체들을 트리 구조로 컬렉션에 추가.
        
        # Set parameters for vision sensors
        self.sim.setObjectInt32Param(self.visionSensors[0], self.sim.visionintparam_entity_to_render, self.collection)
        self.sim.setObjectInt32Param(self.visionSensors[1], self.sim.visionintparam_entity_to_render, self.collection)
        self.sim.setObjectFloatParam(self.visionSensors[0], self.sim.visionfloatparam_far_clipping, self.maxScanDistance)
        self.sim.setObjectFloatParam(self.visionSensors[1], self.sim.visionfloatparam_far_clipping, self.maxScanDistance)
        
        # Add drawing object for lines
        self.red = [1, 0, 0]
        self.lines = self.sim.addDrawingObject(self.sim.drawing_lines, 1, 0, -1, 10000, self.red)

    def cleanup(self):
        self.sim.removeDrawingObject(self.lines)
        print('clear')

    def sensing(self):
        measuredData = []

        # Clear existing lines
        self.sim.addDrawingObjectItem(self.lines, None)

        # Iterate through vision sensors
        for i in range(2):
            if self.sim.readVisionSensor(self.visionSensors[i]) != -1:
                result, detectionState, auxData = self.sim.readVisionSensor(self.visionSensors[i])
                if detectionState:
                    sensorMatrix = self.sim.getObjectMatrix(self.visionSensors[i])
                    relRefMatrix = self.sim.getObjectMatrix(self.self_handle)
                    relRefMatrix = self.sim.getMatrixInverse(relRefMatrix)
                    relRefMatrix = self.sim.multiplyMatrices(relRefMatrix, sensorMatrix)

                    # Prepare transformation data
                    p = [0, 0, 0]
                    p = self.sim.multiplyVector(sensorMatrix, p)
                    t = [p[0], p[1], p[2], 0, 0, 0]

                    # Iterate over vision sensor data
                    for j in range(auxData[2]):
                        for k in range(auxData[1]):
                            index = 2 + 4 * (j * auxData[1] + k)
                            v = [auxData[index+1], auxData[index+2], auxData[index+3], auxData[index+4]]

                            # If generating data, process it
                            if self.generateData:
                                if self.rangeData:
                                    measuredData.append(v[3])  # Store distance data
                                else:
                                    if v[3] < self.maxScanDistance * 0.9999 or not self.discardMaxDistPts:
                                        transformed_p = self.sim.multiplyVector(relRefMatrix, v)
                                        measuredData.append(transformed_p[0])
                                        measuredData.append(transformed_p[1])
                                        measuredData.append(transformed_p[2])

                            # If showing lines, draw them
                            if self.showLines:
                                transformed_p = self.sim.multiplyVector(sensorMatrix, v)
                                t[3], t[4], t[5] = transformed_p[0], transformed_p[1], transformed_p[2]
                                self.sim.addDrawingObjectItem(self.lines, t)
            else:
                pass

        return measuredData
    
    def run_coppelia(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            self.init_coppelia
            self.sensing()
            self.sim.step()
        self.sim.stopSImulation()

# # Simulation setup
# def sysCall_init():
#     lidar = HokuyoLidar()

# def sysCall_cleanup():
#     lidar.cleanup()

# def sysCall_sensing():
#     data = lidar.sensing()
#     # Further processing of `data` as needed.

if __name__ == "__main__":
    client = HokuyoLidar()
    client.init_coppelia()
    client.run_coppelia()
    client.cleanup()