import open3d as o3d
import numpy as np

class FGR:
    def __init__(self, feature_radius=0.025):
        self.feature_radius = feature_radius
        
    def __call__(self, src: np.ndarray, tgt: np.ndarray):
        src_cloud = o3d.geometry.PointCloud()
        tgt_cloud = o3d.geometry.PointCloud()

        src_cloud.points = o3d.utility.Vector3dVector(src)
        tgt_cloud.points = o3d.utility.Vector3dVector(tgt)

        # 计算法向量和曲率
        src_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius * 2, max_nn=30))
        tgt_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius * 2, max_nn=30))
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius * 5, max_nn=100))
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius * 5, max_nn=100))

        # 进行配准
        distance_threshold = self.feature_radius * 1.5
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_cloud, tgt_cloud, src_fpfh, tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        
        R = result.transformation[:3, :3]
        t = result.transformation[:3, 3]
        return R, t
