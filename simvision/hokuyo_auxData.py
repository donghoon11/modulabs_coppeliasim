import numpy as np

auxData = [342.0, 1.0, -4.330127239227295, 0.0, 2.499999761581421, 
 5.0, -4.314690589904785, 0.0, 2.526547908782959, 5.0]
 
 # 데이터 5개씩 묶기
data_points = np.array(auxData).reshape(-1, 5)
print(data_points)

# x, y, z, 거리, 강도 추출
x_vals = data_points[:, 0]
y_vals = data_points[:, 1]
z_vals = data_points[:, 2]
distances = data_points[:, 3]
intensities = data_points[:, 4]

# 결과 확인
print("X 좌표:", x_vals)
print("Y 좌표:", y_vals)
print("Z 좌표:", z_vals)
print("거리:", distances)
print("강도:", intensities)
print(len(data_points))