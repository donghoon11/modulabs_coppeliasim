import math
import sim

def sysCall_init():
    left_wheel = sim.getObjectHandle('Pioneer_p3dx_leftMotor')
    right_wheel = sim.getObjectHandle('Pioneer_p3dx_rightMotor')
    robot_pose = sim.getObjectHandle('robot_pose')
    hokuyo = sim.getObjectHandle('fastHokuyo')
    path = sim.getObjectHandle('Path')
    pose = updateRobotPose()

    wheel_radius = 0.195 / 2
    b = 0.1655
    vref = 0.35
    e = 0.24
    k = 1

    sim.setPathTargetNominalVelocity(path, vref)  # in m/s
    ref_point = sim.getObjectHandle('ref_point')

    laserPoints = []
    # check simGridMap init
    if simGridMap is None:
        print("simGridMap not exists.")
    else:
        simGridMap.init([0.1, 50, 0.9, 0.1, 10, 0.1, 0.1, 0.8, 0.7, 0.3], True)

    simGridMap.init([0.1, 50, 0.9, 0.1, 10, 0.1, 0.1, 0.8, 0.7, 0.3], True)

def sysCall_cleanup():
    simGridMap.release()

def sysCall_actuation():
    wL, wR = None, None
    ptraj, vtraj = getTrajectoryPoint()
    poff, voff = getOffCenterPoint(ptraj, vtraj, e)
    wL, wR = kinematicControl(poff, voff, pose, k)
    sim.setJointTargetVelocity(left_wheel, wL)
    sim.setJointTargetVelocity(right_wheel, wR)

def sysCall_sensing():
    laserPoints = getLaserPoints()
    pose = updateRobotPose()
    simGridMap.updateMapLaser(getLaserPoints(), updateRobotPose())

def getLaserPoints():
    laserScan = sim.callScriptFunction('getMeasuredData@fastHokuyo', sim.scripttype_childscript)
    laserPts = []
    for i in range(0, len(laserScan), 3):
        laserPts.append([laserScan[i], laserScan[i + 1]])
    return laserPts

def getTrajectoryPoint():
    position = sim.getObjectPosition(ref_point, -1)
    orientation = sim.getObjectOrientation(ref_point, -1)
    linear_vel, angular_vel = sim.getObjectVelocity(ref_point)
    if orientation[2] > 0:
        ptraj = [position[0], position[1], orientation[1] - math.pi / 2]
    else:
        ptraj = [position[0], position[1], math.pi / 2 - orientation[1]]
    vtraj = [linear_vel[0], linear_vel[1], angular_vel[2]]

    return ptraj, vtraj

def getOffCenterPoint(ptraj, vtraj, e):
    xc = ptraj[0] + e * math.cos(ptraj[2])
    yc = ptraj[1] + e * math.sin(ptraj[2])
    vxc = vtraj[0] - e * vtraj[2] * math.sin(ptraj[2])
    vyc = vtraj[1] + e * vtraj[2] * math.cos(ptraj[2])
    return [xc, yc], [vxc, vyc]

def kinematicControl(ptraj, vtraj, pose, k):
    ex = ptraj[0] - (pose[0] + e * math.cos(pose[2]))
    ey = ptraj[1] - (pose[1] + e * math.sin(pose[2]))
    vxc = vtraj[0] + k * ex
    vyc = vtraj[1] + k * ey
    wL = (1 / (e * wheel_radius)) * ((e * math.cos(pose[2]) + b * math.sin(pose[2])) * vxc + (e * math.sin(pose[2]) - b * math.cos(pose[2])) * vyc)
    wR = (1 / (e * wheel_radius)) * ((e * math.cos(pose[2]) - b * math.sin(pose[2])) * vxc + (e * math.sin(pose[2]) + b * math.cos(pose[2])) * vyc)
    return wL, wR

def updateRobotPose():
    position = sim.getObjectPosition(robot_pose, -1)
    orientation = sim.getObjectOrientation(robot_pose, -1)
    pose = [position[0], position[1], orientation[2]]
    return pose

