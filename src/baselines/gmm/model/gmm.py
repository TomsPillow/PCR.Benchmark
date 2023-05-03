import open3d as o3d
import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
    def __init__(self, n_components=2, max_iter=20):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def __call__(self, src, tgt):
        # 将numpy数组转换为Open3D点云
        src_cloud = o3d.geometry.PointCloud()
        src_cloud.points = o3d.utility.Vector3dVector(src)
        
        tgt_cloud = o3d.geometry.PointCloud()
        tgt_cloud.points = o3d.utility.Vector3dVector(tgt)
        
        # 使用GMM算法估计点云的分布
        src_gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter).fit(src)
        tgt_gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter).fit(tgt)
        
        # 使用估计的分布对点云进行配准
        src_cluster = src_gmm.predict(src)
        tgt_cluster = tgt_gmm.predict(tgt)
        
        # 获取源点云和目标点云的中心
        src_center = np.mean(src, axis=0)
        tgt_center = np.mean(tgt, axis=0)
        
        # 计算源点云和目标点云的协方差矩阵
        src_cov = np.cov((src - src_center).T)
        tgt_cov = np.cov((tgt - tgt_center).T)
        
        # 使用奇异值分解计算旋转矩阵
        u, _, vh = np.linalg.svd(np.dot(tgt_cov, src_cov.T))
        rotation_matrix = np.dot(u, vh)
        
        # 计算平移向量
        translation_vector = tgt_center - np.dot(rotation_matrix, src_center)
        
        return rotation_matrix, translation_vector
