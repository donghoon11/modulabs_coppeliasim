import math
import numpy as np

T_end = 10
r= 0.5

def curve(t):
    f = 2*math.pi/T_end 
    
    x_ref = x_center + r*math.cos(f*t)
    y_ref = y_center + r*math.sin(f*t)
    
    # x_ref = x_center + r --> x_center = x -r
    # y_ref = y_center --> y_center = y
    
    return x_ref, y_ref
    

def sysCall_init():
    sim = require('sim')

    global joint1
    global joint2
    global endeff
    global endeff_trace
    global x_center
    global y_center
    
    joint1 = sim.getObject("/Joint1")
    joint2 = sim.getObject("/Joint2")
    
    endeff = sim.getObject("/EndEff")
    endeff_trace = sim.addDrawingObject(sim.drawing_linestrip,5,0,-1,100000,[1,0,0])
    
    endeff_position = sim.getObjectPosition(endeff, -1)
    x = endeff_position[0]
    y = endeff_position[1]
    x_center = x - r
    y_center = y

    pass
    
def sysCall_actuation():
    # is executed when simulation is not running
    theta1 = sim.getJointPosition(joint1)   # radian
    theta2 = sim.getJointPosition(joint2)   # radian
    print("theta1 : "+str(theta1)+", theta2 : "+str(theta2))
    
    # dq = Jinv*dr = JInv*(X_ref - X)
    l = 1
    J = np.array([[l*(np.cos(theta1) + np.cos(theta1 + theta2)), l*np.cos(theta1 + theta2)], \
                    [l*(np.sin(theta1) + np.sin(theta1 + theta2)), l*np.sin(theta1 + theta2)]])
    
    Jinv = np.linalg.inv(J)
    
    t = sim.getSimulationTime()
    x_ref, y_ref = curve(t)
    #X_ref = [x_ref, y_ref]
    endeff_position = sim.getObjectPosition(endeff, -1)
    x = endeff_position[0]
    y = endeff_position[1]
    #X = [x, y]
    dr = [x_ref - x, y_ref - y]
    dq = Jinv.dot(dr)
    
    theta1 += dq[0]
    theta2 += dq[1] 
    sim.setJointTargetPosition(joint1, theta1)
    sim.setJointTargetPosition(joint2, theta2)
    
    if t>=T_end:
        sim.stopSimulation()
    
    pass

def sysCall_sensing():
    # is executed before a simulation starts
    
    endeff_position = sim.getObjectPosition(endeff, sim.handle_world)
    sim.addDrawingObjectItem(endeff_trace, endeff_position)
    pass

def sysCall_cleanup():
    # do some clean-up here
    pass

# See the user manual or the available code snippets for additional callback functions and details
