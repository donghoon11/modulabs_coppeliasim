import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class HokuyoLidar:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        # Get handles
        self.visionSensor1Handle = self.sim.getObject("/sensor1")
        self.visionSensor2Handle = self.sim.getObject("/sensor2")
        self.joint1Handle = self.sim.getObject("/joint1")
        self.joint2Handle = self.sim.getObject("/joint2")
        self.sensorRef = self.sim.getObject("/ref")
        
        # Set parameters
        self.maxScanDistance = 5
        self.sim.setObjectFloatParam(self.visionSensor1Handle, 1001, self.maxScanDistance)
        self.sim.setObjectFloatParam(self.visionSensor2Handle, 1001, self.maxScanDistance)
        self.maxScanDistance_ = self.maxScanDistance * 0.9999
        
        # Set scanning angle
        self.scanningAngle = 240 * math.pi / 180
        self.sim.setObjectFloatParam(self.visionSensor1Handle, 1004, self.scanningAngle / 2)
        self.sim.setObjectFloatParam(self.visionSensor2Handle, 1004, self.scanningAngle / 2)
        
        # Configure joints
        self.sim.setJointPosition(self.joint1Handle, -self.scanningAngle / 4)
        self.sim.setJointPosition(self.joint2Handle, self.scanningAngle / 4)

        # Add drawing object for visualization
        self.red = [1, 0, 0]
        self.lines = self.sim.addDrawingObject(self.sim.drawing_lines, 1, 0, -1, 10000, None, None, None, self.red)
        self.showLines = True

    def cleanup(self):
        self.sim.removeDrawingObject(self.lines)

    def sensing(self):
        measuredData = []
        
        if self.not_first_here:
            # Clear existing lines
            self.sim.addDrawingObjectItem(self.lines, None)
            
            # Read data from the sensors
            result1, detectionState1, auxData1 = self.sim.readVisionSensor(self.visionSensor1Handle)
            result2, detectionState2, auxData2 = self.sim.readVisionSensor(self.visionSensor2Handle)

            # Get transformation matrices
            m1 = self.sim.getObjectMatrix(self.visionSensor1Handle)
            m01 = self.sim.getObjectMatrix(self.sensorRef)
            m01 = self.sim.getMatrixInverse(m01)
            m01 = self.sim.multiplyMatrices(m01, m1)
            
            m2 = self.sim.getObjectMatrix(self.visionSensor2Handle)
            m02 = self.sim.getObjectMatrix(self.sensorRef)
            m02 = self.sim.getMatrixInverse(m02)
            m02 = self.sim.multiplyMatrices(m02, m2)
            
            # Process first sensor data
            if auxData1:
                self._process_sensor_data(auxData1, m1, m01, measuredData, self.maxScanDistance_, self.lines)
            
            # Process second sensor data
            if auxData2:
                self._process_sensor_data(auxData2, m2, m02, measuredData, self.maxScanDistance_, self.lines)
        
        self.not_first_here = True
        
        # Send data as signal
        if measuredData:
            data = self.sim.packFloatTable(measuredData)    # return : data: a buffer (values between 0 and 255) that contains packed floating-point numbers
            self.sim.setStringSignal("measuredDataAtThisTime", data)
        else:
            self.sim.clearStringSignal("measuredDataAtThisTime")

    def _process_sensor_data(self, auxData, sensor_matrix, ref_matrix, measured_data, max_distance, lines):
        """Process the sensor data and store or visualize it."""
        for j in range(int(auxData[2])):
            for i in range(auxData[1]):
                w = 2 + 4 * (j * auxData[1] + i)
                v1, v2, v3, v4 = auxData[w+1:w+5]
                
                if v4 < max_distance:
                    p = [v1, v2, v3]
                    p = self.sim.multiplyVector(ref_matrix, p)
                    measured_data.extend(p)

                if self.showLines:
                    p = [v1, v2, v3]
                    p = self.sim.multiplyVector(sensor_matrix, p)
                    t = [0, 0, 0, p[0], p[1], p[2]]
                    self.sim.addDrawingObjectItem(lines, t)

    def run_coppelia(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            self.sensing()
            self.sim.step()
        self.sim.stopSimulation()

if __name__ == "__main__":
    lidar = HokuyoLidar()
    lidar.init_coppelia()
    lidar.run_coppelia()
    lidar.cleanup()
