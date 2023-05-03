import open3d as o3d
import numpy as np

class D3Feat():
    def __init__(self, radius_normal=0.1, radius_feature=0.2, max_nn=100):
        self.radius_normal = radius_normal
        self.radius_feature = radius_feature
        self.max_nn = max_nn

    def __call__(self, src, tgt):
        # 将numpy.array类型的source转换为open3d.geometry.PointCloud类型
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
        # 计算每个点的FPFH局部特征
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=self.max_nn))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=self.max_nn))
        src_fpfh = o3d.registration.compute_fpfh_feature(src, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=self.max_nn))
        tgt_fpfh = o3d.registration.compute_fpfh_feature(tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=self.max_nn))

        # 进行d3feat点云配准
        d3feat_result = o3d.registration.registration_d3feat(src, tgt, self.radius_normal, source_fpfh, target_fpfh)
        transform = np.asarray(d3feat_result.transformation,dtype=np.float32)
        R = transform[:3,:3]
        t = transform[:3,3:].transpose(1,0)[0]
        return R, t
