import open3d as o3d
import numpy as np

class ICP():
    def __init__(self,K=10, R=0.2, D=2, with_ransac=False):
        self.K = K
        self.R = R
        self.D = D
        self.with_ransac = with_ransac
        return

    def __call__(self, src_points, src_normals, tgt_points, tgt_normals):
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_points)
        src_pcd.normals = o3d.utility.Vector3dVector(src_normals)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
        tgt_pcd.normals = o3d.utility.Vector3dVector(tgt_normals)
        init_transform = np.eye(4, dtype=np.float32)
        if self.with_ransac:
            src_pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_pcd,o3d.geometry.KDTreeSearchParamHybrid(radius=self.R, max_nn=self.K))
            tgt_pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_pcd,o3d.geometry.KDTreeSearchParamHybrid(radius=self.R, max_nn=self.K))
            ransac_registration = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(src_pcd,tgt_pcd,src_pcd_fpfh,tgt_pcd_fpfh,
                    True, self.D, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.D)])
            init_transform = np.asarray(ransac_registration.transformation,dtype=np.float32)

        icp_registration = o3d.pipelines.registration.registration_icp(
                    src_pcd, tgt_pcd, self.D, init_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transform = np.asarray(icp_registration.transformation,dtype=np.float32)
        R = transform[:3,:3]
        t = transform[:3,3:].transpose(1,0)[0]
        return R, t