# my_action_interface / action / DockingTwo.action @ odrive input sequence
'''
int32 order
---
int32[] sequence
---
int32[] partial_sequence
'''
# docking.py code

import DockingTwo       # action interface for controling odrive input
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time
from collections import deque

class DockingTwoActionServer(Node):
    def __init__(self):
        super().__init__('dockingtwo_action_server')

        # generate action server
        self.action_server = ActionServer(
            self,
            DockingTwo,
            'dockingtwo',
            self.execute_callback,      # from Node class
            goal_callback=self.goal_callback,   # from Node class
        )

        self.get_logger().info('### DockingTwo Action Server Started')

        # PoseToCmdVel
        self.subscription = self.create_subscription(
            PoseArray,
            '/aruco_poses',
            self.pose_callback,
            1
        )

        # pubilsh 생성
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # pose 초기화
        self.pose_x = 0
        self.pose_y = 0
        self.pose_z = 0
        self.pose_qx = 0
        self.pose_qy = 0
        self.pose_qz = 0
        self.pose_qw = 0

        # PID init
        self.prev_error = 0.0
        self.integral = 0.0
        self.kp = 0.5
        self.integral = 0.0
        self.ki = 0.1
        self.kd = 0.01

        # count
        self.cnt_num = 0
        self.cnt_time = 0
        self.ok = False
        self.again = False

    # 
    async def execute_callback(self, goal_handle):
        pass

    def pose_callback(self, msg):
        pass

    def goal_callback(self, goal_request):
        pass

# cpp 파일과 통신을 위해서 main 함수 호출?
def main(args=None):
    rclpy.init(args=args)

    dockingtwo_action_server = DockingTwoActionServer()
    # dockingtwo_action_client 로부터 request 가 새로 전달받을 때까지 대기.
    rclpy.spin(dockingtwo_action_server)

    dockingtwo_action_server.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()