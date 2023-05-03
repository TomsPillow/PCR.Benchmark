import open3d as o3d
import numpy as np

class FGR:
    def __init__(self):
        pass

    def __call__(self, src, tgt):
        # 将numpy.array类型的source转换为open3d.geometry.PointCloud类型
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
        source_fpfh = o3d.registration.compute_fpfh_feature(src, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        target_fpfh = o3d.registration.compute_fpfh_feature(tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

        # 使用Fast Global Registration算法进行配准
        threshold = 0.25  # 配准阈值
        result = o3d.registration.registration_fast_based_on_feature_matching(
            src, tgt, source_fpfh, target_fpfh,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=threshold))
        transform = np.asarray(result.transformation,dtype=np.float32)
        R = transform[:3,:3]
        t = transform[:3,3:].transpose(1,0)[0]
        return R, t
        