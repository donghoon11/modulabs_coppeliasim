import math
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class ProximitySensorController:
    def __init__(self, sensor_name):
        """
        ProximitySensorController 초기화
        :param sensor_name: CoppeliaSim에서 센서의 경로 또는 이름
        """
        # ZMQ API 클라이언트를 초기화하고 CoppeliaSim에 연결
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        # 센서 핸들 가져오기
        self.sensor_handle = self.sim.getObject(sensor_name)
    
    def read_sensor(self):
        """
        근접 센서 데이터를 읽고, 탐지된 물체까지의 거리를 반환
        :return: 탐지된 물체까지의 거리 (미터 단위)
        """
        result, distanceData, detectedObjectHandle, detectedSurfaceNormalVector = self.sim.readProximitySensor(self.sensor_handle)
        
        if result > 0:
            dx, dy, dz = distanceData
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            return distance
        else:
            return None
    
    def start_sensor_monitoring(self, update_interval=0.05):
        """
        센서 데이터를 주기적으로 읽어서 콘솔에 출력
        :param update_interval: 센서 데이터를 읽는 시간 간격 (초 단위)
        """
        try:
            while self.sim.getSimulationState() != self.sim.simulation_advancing_abouttostop:
                distance = self.read_sensor()
                if distance is not None:
                    print(f'Detected object at distance: {distance:.2f} meters')
                else:
                    print('No object detected')
                
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("Monitoring stopped by user")

# 클래스 사용 예시
if __name__ == '__main__':
    # 센서의 이름 또는 경로를 지정하여 클래스 인스턴스 생성
    sensor_controller = ProximitySensorController('/Proximity_sensor')
    
    # 센서 모니터링 시작
    sensor_controller.start_sensor_monitoring()
