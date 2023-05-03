import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
    def __init__(self, n_components=2, max_iter=20):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def __call__(self, src, tgt):
        # 将numpy数组转换为Open3D点云
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src)
        
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(tgt)
        
        # 使用GMM算法估计点云的分布
        src_gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter).fit(src)
        tgt_gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter).fit(tgt)
        
        # 使用估计的分布对点云进行配准
        src_cluster = src_gmm.predict(src)
        tgt_cluster = tgt_gmm.predict(tgt)
        transformation = self.registration_method(src, tgt)
        src.transform(transformation)
        
        # 将配准后的点云转换回numpy数组
        R = np.asarray(src.get_rotation_matrix())
        t = np.asarray(src.get_translation())
        
        return R, t
