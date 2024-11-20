# slam 기능
# 목적지로 이동
# target 과 현재 위치의 거리가 멀 경우 이동

from grid_mapping import MappingBot
from mcl_coppelia import LocalizationBot
from pathplanning import youBotPP

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time
import matplotlib.pyplot as plt

import logging
import threading
'''
# main 으로 여기서 클래스 실행 통제
# thread 기능 설정.
if __name__ == "__main__":
    planning = youBotPP()
    mapping = MappingBot()

    thread_a = threading.Thread(target=planning.run_coppelia_pp)
    thread_b = threading.Thread(target=mapping.run_coppelia)

    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()
'''

