def sysCall_init():
    global joint_left
    global joint_right
    global point
    global chassis
    global point_trace
    global graph
    global graph_x
    global graph_y
    
    sim = require('sim')


    # do some initialization here
    joint_right = sim.getObject("/Joint_right")
    joint_left = sim.getObject("/Joint_left")
    
    sim.setObjectInt32Param(joint_right, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    sim.setObjectInt32Param(joint_left, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    
    sim.setJointTargetVelocity(joint_left,2)
    sim.setJointTargetVelocity(joint_right,2)
    
    point = sim.getObject("/Point")
    chassis = sim.getObject("/Chassis")
    
    point_trace = sim.addDrawingObject(sim.drawing_linestrip,5,0,-1,100000,[1,0,0])
    
    graph = sim.getObject("/Graph")
    graph_x = sim.addGraphStream(graph, 'x','m',1)
    graph_y = sim.addGraphStream(graph, 'y','m',1)
    sim.addGraphCurve(graph, 'x/y',2,[graph_x, graph_y], [0,0],'m by m',0,[1,0,0],2)
    
    pass

def sysCall_actuation():
    # put your actuation code here
    t = sim.getSimulationTime()
    if (t > 1):
        sim.setJointTargetVelocity(joint_left,100)
    
    point_position = sim.getObjectPosition(point, -1)
    print('x : '+str(point_position[0]) + ', y : '+str(point_position[1]))
    
    eulerAngles = sim.getObjectOrientation(chassis, -1)
    print('angle : '+str(eulerAngles[2]))
    
    sim.addDrawingObjectItem(point_trace, point_position)
    
    sim.setGraphStreamValue(graph, graph_x, point_position[0])
    sim.setGraphStreamValue(graph, graph_y, point_position[1])
    
    pass

def sysCall_sensing():
    # put your sensing code here
    pass

def sysCall_cleanup():
    # do some clean-up here
    pass

# See the user manual or the available code snippets for additional callback functions and details
