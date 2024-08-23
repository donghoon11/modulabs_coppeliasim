import math

FSM_velocity = 0
FSM_position = 1
FSM_damping = 2

def sysCall_init():
    sim = require('sim')
    global joint
    global FSM
    global t_position
    global t_damping
    # do some initialization here
    #
    # Instead of using globals, you can do e.g.:
    # self.myVariable = 21000000
    
    joint = sim.getObject("/Joint")
    
    FSM = FSM_velocity
    t_position = 0
    t_damping = 0


def sysCall_actuation():
    global FSM
    global t_position
    global t_damping
    # put your actuation code here
    # sim.setObjectInt32Param(int objectHandle, int parameterID, int parameter)
    ## sim.jointintparam_dynctrlmode
    '''
    sim.jointdynctrl_free
    sim.jointdynctrl_force
    sim.jointdynctrl_velocity
    sim.jointdynctrl_position
    sim.jointdynctrl_spring
    sim.jointdynctrl_callback
    '''
    # 1) velocity control
    #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    #sim.setJointTargetVelocity(joint, 0.5)
    
    # 2) position control
    #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    #sim.setJointTargetPosition(joint, math.pi/2)    # joint position using radian

    # 3) torque control
    #sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    #thetadot = sim.getJointVelocity(joint)
    #force = -0.5*thetadot       # dissipating energy
    #sim.setJointTargetForce(joint, force)
    
    t = sim.getSimulationTime()
    theta = sim.getJointPosition(joint)
    thetadot = sim.getJointVelocity(joint)
    
    if (FSM==FSM_velocity and theta > -0.5 and theta<0.5):
        FSM = FSM_position
        t_position = t
        print(f'position: {theta}')
        
    if (FSM==FSM_position and t-t_position > 3):
        FSM = FSM_damping
        t_damping = t
        print(f't : {t-t_position}')
        
    if (FSM==FSM_damping and thetadot > -0.1 and thetadot < 0.1 and t-t_damping > 4):
        FSM = FSM_velocity
        # we specify a minimum force at FSM_damping
        # and, that force persists in the memory
        # so we need to clear that force
        sim.setJointTargetForce(joint, 10000)
        print(f'thetadot : {thetadot}, SimulTime : {t}, T_interval : {t-t_damping} ')
    
  
    if (FSM==FSM_velocity):
        sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
        sim.setJointTargetVelocity(joint, 0.5)
    
    if (FSM==FSM_position):
        sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setJointTargetPosition(joint, 0.1)    # joint position using radian    
        
    if (FSM==FSM_damping):
        sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
        
        force = -0.5*thetadot       # dissipating energy
        sim.setJointTargetForce(joint, force)       
    
    pass

def sysCall_sensing():
    # put your sensing code here
    pass

def sysCall_cleanup():
    # do some clean-up here
    pass

# See the user manual or the available code snippets for additional callback functions and details
