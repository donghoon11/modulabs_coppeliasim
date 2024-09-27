import open3d as o3d
import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors

def pcd_show(pointclouds = []):
    show_list = []
    for point_cloud in pointclouds:
        if isinstance(point_cloud, np.ndarray):
            np_point_cloud = point_cloud.reshape((-1, 3))
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(np_point_cloud)
            show_list.append(o3d_point_cloud)
        elif isinstance(point_cloud, o3d.geometry.Geometry3D):
            show_list.append(copy.deepcopy(point_cloud))  # 복사
        else:
            raise ValueError("Unsupported point cloud type.")
    o3d.visualization.draw_geometries(show_list, point_show_normal=False)


def pcd_rotation(point_cloud,roll_deg=0.0,pitch_deg=0.0,yaw_deg=0.0):
    roll_T = np.array([[1,0,0],
                       [0,np.cos(np.deg2rad(roll_deg)),-np.sin(np.deg2rad(roll_deg))],
                       [0,np.sin(np.deg2rad(roll_deg)),np.cos(np.deg2rad(roll_deg))],
                       ])
    pitch_T = np.array([[np.cos(np.deg2rad(pitch_deg)),0,np.sin(np.deg2rad(pitch_deg))],
                       [0,1,0],
                       [-np.sin(np.deg2rad(pitch_deg)),0,np.cos(np.deg2rad(pitch_deg))],
                       ])
    yaw_T = np.array([[np.cos(np.deg2rad(yaw_deg)),-np.sin(np.deg2rad(yaw_deg)),0],
                       [np.sin(np.deg2rad(yaw_deg)),np.cos(np.deg2rad(yaw_deg)),0],
                       [0,0,1],
                       ])
    np_point_cloud = point_cloud.reshape((-1,3))
    # homogeneous matris : np.matmul(np.matmul(yaw_T,pitch_T),roll_T)
    t_pcd = np.matmul(np_point_cloud,np.matmul(np.matmul(yaw_T,pitch_T),roll_T))
    return t_pcd

#1.대응 : 두 포인트가 정렬될 때 최소화되는 것은 거리

### 2중 반복문 사용으로 근접점 찾기
# def find_near_point(source, target):
#     # 최소거리와 비교 매칭된 dist index 리스트들
#     min_dist_list = []
#     min_idx_list= []

#     # source 의 point 하나씩 조회
#     for i, (srcx, srcy, srcz) in enumerate(source):
#         mindist = -1.0
#         minidx = -1.0

#         # target point 하나씩 조회
#         for j, (tgx, tgy, tgz) in enumerate(target):
#             # source 와 target point 간 거리 계산
#             dist = np.linalg.norm(np.array([tgx, tgy, tgz]) - np.array([srcx, srcy, srcz]))

#             if (j == 0):
#                 mindist = dist
#                 minidx = j
#             else:
#                 if(mindist > dist):
#                     mindist = dist
#                     minidx = j

#         min_dist_list.append(mindist)
#         min_idx_list.append(minidx)

#     return np.array(min_dist_list), np.array(min_idx_list)


### nearest neighbors 로 근접점 찾기
def nn(source, target, n_neighbors = 1):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(target)
    distances, indices = neigh.kneighbors(source, return_distance=True)

    return distances.ravel(), indices.ravel()


# 2. transform
### pcd_transform : 포인트 클라우드를 4x4 transform matrix 로 변환하여 반환하는 함수
def pcd_transform(point_cloud, Tm):
    np_point_cloud = point_cloud.reshape((-1,3))
    num = np_point_cloud.shape[0]
    np_point_cloud = np.concatenate([np_point_cloud, np.ones((num, 1))], axis=-1)
    t_pcd = np.dot(Tm, np_point_cloud.T).T
    
    return t_pcd[:,:3]


### find_approximation_transform : 두 포인트클라우드를 이용하여 근사 변환행렬계산하여 반환하는 함수
def find_approximation_transform(source, target):

    A = target.reshape((-1,3))
    B = source.reshape((-1,3))

    # calculate mean
    cp_A = np.mean(A, axis=0).reshape((1,3))
    cp_B = np.mean(B, axis=0).reshape((1,3))

    # find centroid
    X = A - cp_A
    Y = B - cp_B

    # calculate covariance matrix
    D = np.dot(Y.T, X)

    # SVD
    U, S, V_T = np.linalg.svd(D)

    # Rot. matrix
    R = np.dot(V_T, U)

    # 평균점과 회전행렬을 이용하여 이동벡터 계산
    t = cp_A.T - np.dot(R, cp_B.T)

    # 4x4 trans. matrix 로 반환
    Tm = np.eye(4)
    Tm[:3, :3] = R[:3, :3]
    Tm[:3, :3] = t.T

    return Tm


# 3. optimization
def ICP(source, target, iteration=10, threhold = 1e-7):
    Tm = np.eye(4)  # init pose
    error = 0       # init error

    final_Tm = Tm.copy()
    local_source = pcd_transform(source, Tm)

    for _ in range(iteration):
        distances, indices = nn(local_source, target)
        error = distances.mean()        # distances = error between the source point and the target point

        if (error < threhold):
            break
        Tm = find_approximation_transform(local_source, target[indices])
        local_source = pcd_transform(local_source, Tm)
        final_Tm = np.matmul(Tm, final_Tm)

    return error, final_Tm



if __name__ == "__main__":
    corrd = o3d.geometry.TriangleMesh.create_coordinate_frame()
    theta_sample_num = 100
    theta = np.arange(0.0,2*np.pi,(2*np.pi/100))
    r = 1.0
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.zeros_like(x)

    temp_pcd=np.stack([x,y,z],axis=-1)
    temp_pcd2=pcd_rotation(temp_pcd,45.0,0,0)

    #pcd_show([source,target,coord])

    # distances,indices = find_near_point(source,target)
    distances, indices = nn(temp_pcd, temp_pcd2)
    print("near distances :",distances)
    print("near indices :",indices)