from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class DDCar():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

    def init_coppelia(self):
        self.joint_right = self.sim.getObject("/Joint_right")
        self.joint_left = self.sim.getObject("/Joint_left")

        
        # velocity and control mode 
        self.sim.setObjectInt32Param(
            self.joint_right,                       
            self.sim.jointintparam_dynctrlmode,     
            self.sim.jointdynctrl_velocity)
        
        self.sim.setObjectInt32Param(
            self.joint_left, 
            self.sim.jointintparam_dynctrlmode, 
            self.sim.jointdynctrl_velocity)
        
        # point trace
        self.point = self.sim.getObject("/Point")
        self.chassis = self.sim.getObject("/Chassis")

        self.point_trace = self.sim.addDrawingObject(
            self.sim.drawing_linestrip,5,0,-1,100000,[1,0,0])
    
        # self.graph = self.sim.getObject("/Graph")
        # self.graph_x = self.sim.addGraphStream(self.graph, 'x','m',1)
        # self.graph_y = self.sim.addGraphStream(self.graph, 'y','m',1)


    def sysCall_actuation(self):
        # put your actuation code here
        self.sim.setJointTargetVelocity(self.joint_left, 2)
        self.sim.setJointTargetVelocity(self.joint_right, 2)

        if self.sim.getSimulationTime() > 2:
            self.sim.setJointTargetVelocity(self.joint_right,20)
        
        self.point_position = self.sim.getObjectPosition(self.point, -1)
        print('x : '+str(self.point_position[0]) + ', y : '+str(self.point_position[1]))
        
        self.eulerAngles = self.sim.getObjectOrientation(self.chassis, -1)
        print('angle : '+str(self.eulerAngles[2]))
        
        self.sim.addDrawingObjectItem(self.point_trace, self.point_position)
        
        # REMOTE API 로 그래프 그리면 coppeliasim 화면 상에서 업데이트되는 모습.
        # self.sim.addGraphCurve(
        #     self.graph, 'x/y',2,
        #     [self.graph_x, self.graph_y], [0,0],'m by m',
        #     0,[1,0,0],2)
        
        # self.sim.setGraphStreamValue(self.graph, self.graph_x, self.point_position[0])
        # self.sim.setGraphStreamValue(self.graph, self.graph_y, self.point_position[1])
        
    def run_coppelia(self):
        # start simulation
        self.sim.startSimulation()
        while True :
            # run the sysCall_actuation()
            self.sysCall_actuation()
            print(f'Joint_left : {self.sim.getJointVelocity(self.joint_left)}')
            print(f'Joint_right : {self.sim.getJointVelocity(self.joint_right)}')
            print('')
        # self.sim.stopSimulation()

if __name__ == "__main__":
    client = DDCar()
    client.init_coppelia()
    client.run_coppelia()
