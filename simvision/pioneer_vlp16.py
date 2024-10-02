from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Pioneer:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.simVision = self.client.require("simVision")
        self.run_flag = True


    def init_coppelia(self):
        self.visionSensorHandles = []
        self.ptCloudHandle = self.sim.getObject("/ptCloud")

        for i in range(4):
            self.visionSensorHandles.append(self.sim.getObject(f"/sensor[{i}]"))

        self.frequency = 5  # 5 Hz
        self.options = 2 + 8  # 옵션 설정
        self.pointSize = 2
        self.coloring_closeAndFarDistance = [1, 4]
        self.displayScaling = 0.999  # 점들이 객체 내에서 사라지지 않도록 스케일 조정

        self.h = self.simVision.createVelodyneVPL16(self.visionSensorHandles, 
                                                    self.frequency, 
                                                    self.options, 
                                                    self.pointSize, 
                                                    self.coloring_closeAndFarDistance, 
                                                    self.displayScaling, 
                                                    self.ptCloudHandle)
        self.printInterval = 1
        self.lastPrintTime = 0
        self.ptCloud = None


    def sysCall_sensing(self):
        # Velodyne 센서 데이터를 절대 좌표계에서 처리
        data = self.simVision.handleVelodyneVPL16(self.h + self.sim.handleflag_abscoords, 
                                                  self.sim.getSimulationTimeStep())
        # 포인트 클라우드 갱신 및 표시
        if self.ptCloud:
            self.sim.removePointsFromPointCloud(self.ptCloud, 0, None, 0)
        else:
            self.ptCloud = self.sim.createPointCloud(0.02, 20, 0, self.pointSize)  # 포인트 클라우드 생성
        self.sim.insertPointsIntoPointCloud(self.ptCloud, 0, data)

        # data per 1sec
        currentTime = self.sim.getSimulationTime()
        if currentTime - self.lastPrintTime >= self.printInterval:
            self.printLidarData(data)
            self.lastPrintTime = currentTime


    def printLidarData(self, data):
        print("Velodyne VPL-16 Data Sample:")
        print("Number of points: {}".format(len(data) // 3))
        # first 15 points
        for i in range(0, min(15, len(data)), 3):
            print("Point {}: ({:.3f}, {:.3f}, {:.3f})".format(
                (i // 3) + 1, data[i], data[i + 1], data[i + 2]))
        print("--------------------")


    def sysCall_cleanup(self):
        self.simVision.destroyVelodyneVPL16(self.h)


    def run_coppelia(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while self.run_flag:
            # self.run_step()
            self.sysCall_sensing()
            self.sim.step()
        self.sim.stopSimulation()

if __name__ == "__main__":
    client = Pioneer()
    client.init_coppelia()
    client.run_coppelia()
