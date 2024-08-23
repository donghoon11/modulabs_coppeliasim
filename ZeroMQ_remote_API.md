```python
from coppeliasim_zmqremoteapi_cient import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 3:
    print(f'Simulation time: {t:.2f} [s]')
    sim.step()
sim.stopSimulation()
```

### stepping mode

기본적으로, CoppeliaSim은 자동으로 하나의 시뮬레이션 스텝을 실행한 후 다음 스텝을 실행하는 방식으로 시뮬레이션을 실행. 

시뮬레이션 스텝을 수동으로 트리거해야 하는 상황에서 스텝 모드를 사용할 수 있다. 외부 애플리케이션에서 스텝 모드를 활성화하고 각 개별 스텝을 트리거하기 위한 전용 함수가 제공된다.