import numpy as np
import open3d as o3d

def transform(pc, R, t):
    '''
    pc: 3 x N
    R: 3 x 3
    t: 3 x 1
    '''
    return np.matmul(R, pc,dtype=np.float32) + t.reshape(3,1)


def features_project(features1, features2):
    return np.matmul(features1, features2)

def decentralization(pc):
    heart = np.mean(pc, axis=0)
    dc_pc = pc - heart
    return dc_pc

def vec_softmax(f):
    f -= np.max(f)
    x = np.exp(f) / np.sum(np.exp(f))
    return x

def src_softmax(deep_graph):
    for i in range(len(deep_graph)):
        deep_graph[i] = vec_softmax(deep_graph[i])
    return deep_graph

def tgt_softmax(deep_graph):
    deep_graph = deep_graph.transpose(1,0)
    for j in range(len(deep_graph)):
        deep_graph[j] = vec_softmax(deep_graph[j])
    return deep_graph.transpose(1,0)

def src_correspondence_fliter(deep_graph, correlation_thresh=0.5):
    src_correspondence = set()
    for i in range(len(deep_graph)):
        max_correlation_idx = np.argmax(deep_graph[i])
        max_correlation = deep_graph[i][max_correlation_idx]
        if max_correlation > correlation_thresh:
            src_correspondence.add(tuple((i, max_correlation_idx)))
    return src_correspondence

def tgt_correspondence_fliter(deep_graph, correlation_thresh=0.5):
    deep_graph = deep_graph.transpose(1,0)
    tgt_correspondence = set()
    for j in range(len(deep_graph)):
        max_correlation_idx = np.argmax(deep_graph[j])
        max_correlation = deep_graph[j][max_correlation_idx]
        if max_correlation > correlation_thresh:
            tgt_correspondence.add(tuple((max_correlation_idx, j)))
    return tgt_correspondence


def estimate_Rt(src_pc, tgt_pc, correspondences):
    src_dc_pc = decentralization(src_pc)
    tgt_dc_pc = decentralization(tgt_pc)
    H = np.zeros((3,3))
    for pair in correspondences:
        p = src_dc_pc[pair[0]].reshape(-1,1)
        q = tgt_dc_pc[pair[1]].reshape(-1,1)
        H = H + np.matmul(p, q.T)

    u, _, vt = np.linalg.svd(H, full_matrices=True)
    R = np.matmul(vt.T,u.T)
    translation = tgt_pc - rotation(src_pc.transpose(1,0), R).transpose(1,0)
    t = np.mean(translation,axis=0)
    return R, t

def point_cloud_registration(src_points, src_normals, tgt_points, tgt_normals, K=32, M=18, correlation_thresh=0.5):
    src_knns_idx = search_knns(src_points, K)
    tgt_knns_idx = search_knns(tgt_points, K)
    m_idx = np.random.choice(K, M, replace=False)
    src_mns_idx = src_knns_idx[:,m_idx]
    tgt_mns_idx = tgt_knns_idx[:,m_idx]
    src_feat = calculate_PPFs(src_points, src_normals, src_mns_idx).transpose((1,0,2))     # src_feat: K x points_num x 4
    tgt_feat = calculate_PPFs(tgt_points, tgt_normals, tgt_mns_idx).transpose((1,0,2))     # tgt_feat: K x points_num x 4

    avg_deep_graph = np.zeros((src_feat.shape[1], tgt_feat.shape[1]))
    for i in range(M):
        deep_graph = features_project(src_feat[i], tgt_feat[i].transpose(1,0))
        avg_deep_graph = avg_deep_graph + deep_graph
    avg_deep_graph = avg_deep_graph / M

    correspondences = set()
    least = 128
    while len(correspondences)<least:
        # source correspondence fliter
        src_correspondence = src_correspondence_fliter(avg_deep_graph, correlation_thresh=correlation_thresh)

        # target correspondence fliter
        tgt_correspondence = tgt_correspondence_fliter(avg_deep_graph, correlation_thresh=correlation_thresh)

        # matched correspondences
        correspondences = src_correspondence.intersection(tgt_correspondence)
        correlation_thresh = correlation_thresh * 0.99
        least = int(least * 0.99)

    R, t = estimate_Rt(src_points, tgt_points, correspondences)
    return R, t, correspondences, avg_deep_graph

def vecs_angle(vec1, vec2):
    value = np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if value > 1:
        return np.arccos(1)
    elif value < -1:
        return np.arccos(-1)
    return np.arccos(value)

def calculate_PPFs(points, normals, nns_idx):
    PPFs = []
    for i in range(len(points)):
        x = points[i]
        x_normal = normals[i]
        nns = nns_idx[i]
        PPF = []
        for n_idx in nns:
            nn = points[n_idx]
            nn_normal = normals[n_idx]
            delta = x - nn
            # value = np.asarray([vecs_angle(x_normal,delta) / np.pi, vecs_angle(nn_normal, delta) / np.pi, vecs_angle(x_normal,nn_normal) / np.pi, np.linalg.norm(delta)])
            value = np.asarray([vecs_angle(x_normal,delta), vecs_angle(nn_normal, delta), vecs_angle(x_normal,nn_normal), np.linalg.norm(delta)])
            value = value / np.linalg.norm(value)   # unitization
            PPF.append(value)
        PPFs.append(np.asarray(PPF))
    return np.asarray(PPFs)

def search_knns(points, K):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_kd_tree = o3d.geometry.KDTreeFlann(pcd)
    nns = []
    for i in range(len(pcd.points)):
        xyz = pcd.points[i]
        [_,idx,_] = pcd_kd_tree.search_knn_vector_3d(xyz, K+1)
        idx = np.asarray(idx)
        nns.append(idx[1:])
    return np.asarray(nns)  # return K nearest neighbors' index in pcd.points

class DGM:
    def __init__(self,K=32,M=18,correlation_thresh=0.5):
        self.K = K
        self.M = M
        self.correlation_thresh = correlation_thresh
    def __call__(self,src_points,src_normals,tgt_points,tgt_normals):
        return point_cloud_registration(src_points,src_normals,tgt_points,tgt_normals,self.K,self.M,correlation_thresh=self.correlation_thresh)
