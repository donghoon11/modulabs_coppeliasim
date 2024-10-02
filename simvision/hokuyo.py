import sim
import numpy as np

class HokuyoLidar:
    def __init__(self):
        self.maxScanDistance = 5
        self.showLines = True
        self.generateData = True
        self.rangeData = True  # if False, then X/Y/Z data rel to sensor base
        self.discardMaxDistPts = True
        
        # Get sensor and object handles
        self.self_handle = sim.simxGetObjectHandle(sim.simxGetObjectHandle(".."))
        self.visionSensors = [
            sim.simxGetObjectHandle(sim.simxGetObjectHandle("../sensor1")),
            sim.simxGetObjectHandle(sim.simxGetObjectHandle("../sensor2"))
        ]
        
        # Create a collection
        self.collection = sim.simxCreateCollection(0)
        sim.simxAddItemToCollection(self.collection, sim.sim_handle_all, -1, 0)
        sim.simxAddItemToCollection(self.collection, sim.sim_handle_tree, self.self_handle, 1)
        
        # Set parameters for vision sensors
        sim.simxSetObjectInt32Param(self.visionSensors[0], sim.sim_visionintparam_entity_to_render, self.collection)
        sim.simxSetObjectInt32Param(self.visionSensors[1], sim.sim_visionintparam_entity_to_render, self.collection)
        sim.simxSetObjectFloatParam(self.visionSensors[0], sim.sim_visionfloatparam_far_clipping, self.maxScanDistance)
        sim.simxSetObjectFloatParam(self.visionSensors[1], sim.sim_visionfloatparam_far_clipping, self.maxScanDistance)
        
        # Add drawing object for lines
        self.red = [1, 0, 0]
        self.lines = sim.simxAddDrawingObject(sim.sim_drawing_lines, 1, 0, -1, 10000, self.red)

    def cleanup(self):
        sim.simxRemoveDrawingObject(self.lines)

    def sensing(self):
        measuredData = []

        # Clear existing lines
        sim.simxAddDrawingObjectItem(self.lines, None)

        # Iterate through vision sensors
        for i in range(2):
            result, detectionState, auxData = sim.simxReadVisionSensor(self.visionSensors[i])

            if detectionState:
                sensorMatrix = sim.simxGetObjectMatrix(self.visionSensors[i])
                relRefMatrix = sim.simxGetObjectMatrix(self.self_handle)
                relRefMatrix = sim.simxGetMatrixInverse(relRefMatrix)
                relRefMatrix = sim.simxMultiplyMatrices(relRefMatrix, sensorMatrix)

                # Prepare transformation data
                p = [0, 0, 0]
                p = sim.simxMultiplyVector(sensorMatrix, p)
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
                                    transformed_p = sim.simxMultiplyVector(relRefMatrix, v)
                                    measuredData.append(transformed_p[0])
                                    measuredData.append(transformed_p[1])
                                    measuredData.append(transformed_p[2])

                        # If showing lines, draw them
                        if self.showLines:
                            transformed_p = sim.simxMultiplyVector(sensorMatrix, v)
                            t[3], t[4], t[5] = transformed_p[0], transformed_p[1], transformed_p[2]
                            sim.simxAddDrawingObjectItem(self.lines, t)

        return measuredData


# Simulation setup
def sysCall_init():
    lidar = HokuyoLidar()

def sysCall_cleanup():
    lidar.cleanup()

def sysCall_sensing():
    data = lidar.sensing()
    # Further processing of `data` as needed.
