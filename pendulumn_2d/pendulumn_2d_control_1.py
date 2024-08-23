import numpy as np
import matplotlib.pyplot as plt

# 시작점과 끝점 정의
start_point = np.array([1, 1])
end_point = np.array([0, -1.5])

# 속도 (m/s)
velocity = 0.5

# 시간 간격 (10 ms)
time_interval = 0.01

# 두 점 사이의 거리 계산
distance = np.linalg.norm(end_point - start_point)

# 전체 이동 시간 계산
total_time = distance / velocity

# 시간 벡터 생성
time_vector = np.arange(0, total_time, time_interval)

# 각 시간에서의 위치를 저장할 배열 초기화
positions = np.zeros((len(time_vector), 2))

# 방향 벡터 계산
direction_vector = (end_point - start_point) / distance

# 각 시간에서의 위치 계산
for i, t in enumerate(time_vector):
    current_position = start_point + velocity * t * direction_vector
    positions[i, :] = current_position

# 결과 출력
print('Time (s)     X         Y')
for i in range(len(time_vector)):
    print(f'{time_vector[i]:.2f}     {positions[i, 0]:.2f}     {positions[i, 1]:.2f}')

# 위치 그래프 그리기
plt.figure()
plt.plot(positions[:, 0], positions[:, 1], '-o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Movement from (1,1) to (0,-1.5)')
plt.grid(True)
plt.show()

# 링크 길이 정의
L1 = 1.0
L2 = 0.75

# 역기구학 계산 함수 정의
def inverse_kinematics(x, y, L1, L2):
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)  # 두 개의 솔루션 중 하나 선택
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

# 조인트 각도 초기화
joint_angles = np.zeros((len(time_vector), 2))

# 각 목표 위치에서의 조인트 각도 계산
for i in range(len(time_vector)):
    x = positions[i, 0]
    y = positions[i, 1]
    theta1, theta2 = inverse_kinematics(x, y, L1, L2)
    joint_angles[i, :] = [theta1, theta2]

# 조인트 각도 플롯
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time_vector, joint_angles[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Theta1 (rad)')
plt.title('Joint Angle Theta1')

plt.subplot(2, 1, 2)
plt.plot(time_vector, joint_angles[:, 1])
plt.xlabel('Time (s)')
plt.ylabel('Theta2 (rad)')
plt.title('Joint Angle Theta2')

plt.show()

# 작업 공간 내 경로 플롯
plt.figure()
plt.plot(positions[:, 0], positions[:, 1], '-o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('End Effector Path')
plt.grid(True)
plt.show()

# 조인트 트래젝터리 플롯
plt.figure()
for i in range(0, len(time_vector), 5):
    theta1 = joint_angles[i, 0]
    theta2 = joint_angles[i, 1]
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    plt.plot([0, x1, x2], [0, y1, y2], '-o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2-Link Manipulator Trajectory')
plt.grid(True)
plt.show()
