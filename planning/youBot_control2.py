from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

class youBotControl():
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.run_flag = True

    def init_coppelia(self):
        self.vehicle_reference = self.sim.getObject('/youBot_ref')
        self.target = self.sim.getObject('/goalDummy')
        # self.sim.setObjectPosition(self.target, [0,0,0], self.sim.handle_parent)
        # self.sim.setObjectOrientation(self.target, [0,0,0], self.sim.handle_parent)
        # self.sim.setObjectParent(self.target, -1, True)

        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),  # front left
            self.sim.getObject('/rollingJoint_rl'),  # rear left
            self.sim.getObject('/rollingJoint_fr'),   # front right
            self.sim.getObject('/rollingJoint_rr'),  # rear right
        ]

        self.prev_forwback_vel = 0
        self.prev_side_vel = 0
        self.prev_rot_vel = 0

    def actuation(self):
        # vehicle_reference 에 대한 target의 상대 위치
        rel_p = self.sim.getObjectPosition(self.target, self.vehicle_reference)
        rel_e = self.sim.getObjectOrientation(self.target, self.vehicle_reference)
        p_parm = 20
        max_v = 2
        p_parm_rot = 10
        max_v_rot = 3
        accel_f = 0.035

        forwback_vel = rel_p[1] * p_parm
        side_vel = rel_p[0] * p_parm
        v = (forwback_vel**2 + side_vel**2)**0.5
        if v > max_v:
            forwback_vel *= max_v / v
            side_vel *= max_v / v

        rot_vel = -rel_e[2] * p_parm_rot
        if abs(rot_vel) > max_v_rot :
            rot_vel = max_v_rot * rot_vel / abs(rot_vel)

        df = forwback_vel - self.prev_forwback_vel
        ds = side_vel - self.prev_side_vel
        dr = rot_vel - self.prev_rot_vel

        if abs(df) > max_v * accel_f:
            df = max_v * accel_f * df / abs(df)
        if abs(ds) > max_v * accel_f:
            ds = max_v * accel_f * ds / abs(ds)
        if abs(dr) > max_v_rot * accel_f:
            dr = max_v_rot * accel_f * dr / abs(dr)

        forwback_vel = self.prev_forwback_vel + df
        side_vel = self.prev_side_vel + ds
        rot_vel = self.prev_rot_vel + dr

        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
        
        self.prev_forwback_vel = forwback_vel
        self.prev_side_vel = side_vel
        self.prev_rot_vel = rot_vel

    def check(self):
            symbol = False
            p1 = self.sim.getObjectPosition(self.target)
            p2 = self.sim.getObjectPosition(self.vehicle_reference)
            p_error = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            o_error = abs(self.sim.getObjectOrientation(self.vehicle_reference, self.target)[2])
            print(f'p_error : {p_error} | o_error : {o_error}')
            print()
            # stop 조건을 더 타이트하게
            if p_error < 0.7 and o_error < 0.00055:
                '''
                몇 가지 조건 추가.
                '''
                symbol = True
            return symbol

    def run_coppelia(self):
        self.sim.startSimulation()
        while self.run_flag:
            if self.check():
                break
            self.actuation()
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    controller = youBotControl()
    controller.init_coppelia()
    controller.run_coppelia()