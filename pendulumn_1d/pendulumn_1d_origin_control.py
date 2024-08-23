import math

def sysCall_init():
    sim = require('sim')
    global joint
    # do some initialization here
    #
    # Instead of using globals, you can do e.g.:
    # self.myVariable = 21000000
    
    joint = sim.getObject("/Joint")


def sysCall_actuation():
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
    sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    thetadot = sim.getJointVelocity(joint)
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
