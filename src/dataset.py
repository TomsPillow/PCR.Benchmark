import open3d as o3d
import h5py as h5
from utils.generator import generate_random_rotation_matrix, generate_random_tranlation_vector, jitter_point_cloud
from utils.tools import shuffle,estimate_normals
from utils.transform import transform


def search_knns(points, K=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_kd_tree = o3d.geometry.KDTreeFlann(pcd)
    nns = []
    for i in range(len(pcd.points)):
        xyz = pcd.points[i]
        [_,idx,_] = pcd_kd_tree.search_knn_vector_3d(xyz, K+1)
        idx = np.asarray(idx)
        nns.append(idx[1:])
    return np.asarray(nns, dtype=np.int32)  # return K nearest neighbors' index in pcd.points: N x K


class dataset():
    def __init__(self, random_n=-1, h5fs_list=None, K=10, noise=False, sigma=0.1, clip=0.05):
        self.K = K
        self.sigma = sigma
        self.clip = clip
        self.noise = noise
        self.xyzs = []      # model_num x points_num x 3
        for h5f in h5fs_list:
            f = h5.File(h5f)
            self.xyzs.append(np.asarray(f['data'],dtype=np.float32))
        self.xyzs = np.asarray(self.xyzs,dtype=object)
        self.xyzs = np.concatenate(self.xyzs,axis=0)
        self.xyzs = np.asarray(self.xyzs,dtype=np.float32)
        if random_n < self.xyzs.shape[0] and random_n > 0:
            ids = np.random.choice(self.xyzs.shape[0],random_n,replace=False)
            self.xyzs=self.xyzs[ids]
        
    def __getitem__(self, index):
        gt_R = generate_random_rotation_matrix()
        gt_t = generate_random_tranlation_vector()
        src_points = self.xyzs[index]
        shuffed_idx, tgt_points = shuffle(src_points)
        tgt_points = transform(tgt_points.transpose(1,0), gt_R, gt_t).transpose(1,0)
        if self.noise:
            tgt_points = jitter_point_cloud(tgt_points,sigma=self.sigma,clip=self.clip)
        src_normals = estimate_normals(src_points)
        tgt_normals = estimate_normals(tgt_points)
        tgt_knns = search_knns(tgt_points, K=self.K)
        src_knns = shuffed_idx[tgt_knns]
        return src_points.astype(np.float32), src_normals.astype(np.float32), src_knns, \
            tgt_points.astype(np.float32), tgt_normals.astype(np.float32), tgt_knns, \
                gt_R.astype(np.float32), gt_t.astype(np.float32)

    def __len__(self):
        return self.xyzs.shape[0]