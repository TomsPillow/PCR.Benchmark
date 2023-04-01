import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_vecs_angle(vecs1:torch.Tensor, vecs2:torch.Tensor) -> torch.Tensor:
    '''__summary__
        caculate vectors angles in batches

    Args:
        vecs1 (torch.Tensor): [B, N, K, C]
        vecs2 (torch.Tensor): [B, N, K, C]

    Returns:
        torch.Tensor:[B, N, K, 1]
    '''
    projects = torch.sum(vecs1 * vecs2, dim=-1, keepdim=True)   # [B, N, K, 1]
    norm1 = torch.norm(vecs1, dim=-1, keepdim=True) # [B, N, K, 1]
    norm2 = torch.norm(vecs2, dim=-1, keepdim=True) # [B, N, K, 1]
    angles = torch.arccos(projects / norm1 / norm2)
    return angles


def batch_calcuate_PPF(points:torch.Tensor, normals:torch.Tensor, knns:torch.Tensor) -> torch.Tensor:
        '''__summary__
            calculate point pair features(PPF) in batches

        Args:
            points (torch.Tensor): points in cloud, [B, N, C]
            normals (torch.Tensor): normals of points in cloud, [B, N, C]
            knns (torch.Tensor): K nearest neighbors indices of each point in cloud, [B, N, K]
        
        Returns:
            torch.Tensor:[B, N, K, C]
        '''
        B = points.shape[0]
        N = points.shape[1]
        K = knns.shape[2]
        C = points.shape[2]
        knns_xyz = points[:, knns][:,0]             # [B, N, K, C]
        knns_normals_xyz = normals[:, knns][:,0]     # [B, N, K, C]
        points_xyz = points.repeat(1,K,1).reshape(B, N, K, C)
        normals_xyz = normals.repeat(1,K,1).reshape(B, N, K, C)
        deltas = knns_xyz - points_xyz                   # [B, N, K, C]
        ppf1 = batch_vecs_angle(normals_xyz, deltas)
        ppf2 = batch_vecs_angle(knns_normals_xyz, deltas)
        ppf3 = batch_vecs_angle(normals_xyz, knns_normals_xyz)
        ppf4 = torch.norm(deltas, dim=-1, keepdim=True)
        PPF = torch.concat([ppf1,ppf2,ppf3,ppf4],dim=-1)
        PPF = torch.where(torch.isnan(PPF), torch.full_like(PPF, 0), PPF)
        return PPF

def batch_calcuate_deltas(points:torch.Tensor, knns:torch.Tensor):
    '''__summary__
            calculate delta vectors between points in cloud and their neighbors in batches

        Args:
            points (torch.Tensor): points in cloud, [B, N, C]
            knns (torch.Tensor): K nearest neighbors indices of each point in cloud, [B, N, K]
        
        Returns:
            torch.Tensor:[B, N, K, C]
    '''
    B = points.shape[0]
    N = points.shape[1]
    K = knns.shape[2]
    C = points.shape[2]
    knns_xyz = points[:, knns][:,0]             # [B, N, K, C]
    points_xyz = points.repeat(1,K,1).reshape(B, N, K, C)
    deltas = knns_xyz - points_xyz              # [B, N, K, C]
    return deltas

def KNN(inp, k = 20):
    """
        Input Tensor : (B, F, N)
        1. Does KNN and return (B, F, N, K) features
        2. Concat to (B, 2F, N, K) -> Broadcasted Input tensor to (B, 2F, N, K) for concat
    """
    B, F, N = inp.shape[:3]
    inp = inp.view(B, F, N)
    nns = inp.transpose(1, 2) # nns : (B, N, F) 

    # Doing KNN (B, F, N) -> (B, F, K, N)
    d = torch.cdist(nns, nns, p = 2.0, compute_mode = 'use_mm_for_euclid_dist') # d: (B, N, N)
    _, idx = torch.topk(input = d, k = k, dim = -1, largest = False) # idx : (B, N, K)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, F) # idx : (B, N, K, F)

    nns = nns.unsqueeze(2).expand(-1, -1, k, -1) # nns : (B, N, K, F)
    nns = torch.gather(nns, 1, idx) # nns : (B, N, K, F)

    return torch.cat([nns.permute(0, 3, 1, 2), inp.unsqueeze(-1).repeat(1, 1, 1, k)], 1) # (B, 2F, N, K)


class LocalFeatureExtractorLayer(nn.Module):
    def __init__(self, in_channels=10):
        super().__init__()
        self.conv1 = nn.Conv2d(2*in_channels, 128, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(256, 128, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(512, 512, 1, 1, 0)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, input):        # input shape: batch_size x channels_num x point_num x K
        fs = []
        x = F.relu(self.bn1(self.conv1(KNN(input))))          # 20 -> 128
        x,_ = torch.max(x, dim=-1, keepdim=True)              # [B, C, N, 1], C = 128
        fs.append(x)

        x = F.relu(self.bn2(self.conv2(KNN(x))))              # 128 -> 128
        x,_ = torch.max(x, dim=-1, keepdim=True)              # [B, C, N, 1], C = 128
        fs.append(x)

        x = F.relu(self.bn3(self.conv3(KNN(x))))              # 128 -> 256
        x,_ = torch.max(x, dim=-1, keepdim=True)              # [B, C, N, 1], C = 256
        fs.append(x)

        x = F.relu(self.bn4(self.conv4(KNN(x))))              # 256 -> 512
        x,_ = torch.max(x, dim=-1, keepdim=True)              # [B, C, N, 1], C = 512
        fs.append(x)

        fs = torch.cat(fs, dim=1)                                    # [B, embeded_C, N, 1], embeded_C = 1024 -> 1024
        output = F.relu(self.bn5(self.conv5(fs))).squeeze(dim=-1)    # [B, C, N], C = 1024
        return output

class AffinityMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, FP, FQ):
        A = torch.matmul(FQ.permute(0,2,1), FP)
        return A

class SinkhornLayer(nn.Module):
    def __init__(self, iter_n=5):
        super().__init__()
        self.iter_n = iter_n
    def forward(self, log_alpha):
        for _ in range(self.iter_n):
            # row normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            # column normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        return torch.exp(log_alpha)

def uncentralization(pc):
    center = torch.mean(pc, dim=1)
    center = torch.unsqueeze(center, dim=1).repeat(1, pc.shape[1], 1)
    return pc-center

# class CorrespondencesExtractionWithTransformEstimation():
#     def __init__(self,correlation_thresh=0.5):
#         self.correlation_thresh = correlation_thresh

#     def correspondences_extraction(self, M):
#         t2s_M = torch.where(M<torch.max(M,dim=2,keepdim=True)[0],0,M)
#         s2t_M = torch.where(M<torch.max(M,dim=1,keepdim=True)[0],0,M)
#         t2s_M = torch.where(t2s_M<self.correlation_thresh, 0., 1.)
#         s2t_M = torch.where(s2t_M<self.correlation_thresh, 0., 1.)
#         correspondences_M = t2s_M * s2t_M
#         return correspondences_M

#     def __call__(self, Ppc, Qpc, M):
#         correspondences_M = self.correspondences_extraction(M)
#         matched_Ppc = correspondences_M @ Ppc
#         uc_matched_Ppc = uncentralization(matched_Ppc)
#         uc_Qpc = uncentralization(Qpc)
#         S = uc_matched_Ppc.permute(0,2,1) @ uc_Qpc
#         U,_,VT = torch.linalg.svd(S, full_matrices=True)
#         R=VT.permute(0,2,1) @ U.permute(0,2,1)
#         t=torch.mean(Qpc.permute(0,2,1)-R @ Ppc.permute(0,2,1),dim=2)
#         return correspondences_M, R, t


class TransformationEstimation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Ppc, Qpc, M):
        matched_Ppc = M @ Ppc
        uc_matched_Ppc = uncentralization(matched_Ppc)
        uc_Qpc = uncentralization(Qpc)
        S = uc_matched_Ppc.permute(0,2,1) @ uc_Qpc
        U,_,VT = torch.linalg.svd(S, full_matrices=True)
        R = VT.permute(0,2,1) @ U.permute(0,2,1)
        t = torch.mean(Qpc.permute(0,2,1) - R @ Ppc.permute(0,2,1),dim=2)
        return R, t

class DGMNet(nn.Module):
    def __init__(self, K=10):
        super().__init__()
        self.K = K
        self.LFE = LocalFeatureExtractorLayer(in_channels=K*10)
        self.AML = AffinityMatrixLayer()
        self.IN = nn.InstanceNorm2d(num_features=1)
        self.SL = SinkhornLayer()
        self.TE = TransformationEstimation()

    def forward(self, Ppc, Pn, Pknns, Qpc, Qn, Qknns):
        '''_summary_

        Args:
            Ppc (torch.Tensor): points in source point cloud, [B, N, C]
            Pn (torch.Tensor): normals of source point cloud, [B, N, C]
            Pknns (torch.Tensor): K nearest neighbors' indices of points in source point cloud, [B, N, K]
            Qpc (torch.Tensor): points in target point cloud, [B, N, C]
            Qn (torch.Tensor): normals of target point cloud, [B, N, C]
            Qknns (torch.Tensor): K nearest neighbors' indices of points in target point cloud, [B, N, K]

        '''
        B = Ppc.shape[0]
        N = Ppc.shape[1]
        K = Pknns.shape[2]
        Pf1 = Ppc.unsqueeze(dim=-2).repeat(1,1,K,1)
        Qf1 = Qpc.unsqueeze(dim=-2).repeat(1,1,K,1)
        Pf2 = batch_calcuate_deltas(Ppc,Pknns)
        Pf3 = batch_calcuate_PPF(Ppc,Pn,Pknns)
        Qf2 = batch_calcuate_deltas(Ppc,Pknns)
        Qf3 = batch_calcuate_PPF(Qpc,Qn,Qknns)
        Pf = torch.concat((Pf1,Pf2,Pf3),dim=-1).reshape(B,N,K*10).permute(0,2,1).contiguous() # [B, N, K, C] -> [B, N, K*C], C = 10 -> [B, F, N]
        Qf = torch.concat((Qf1,Qf2,Qf3),dim=-1).reshape(B,N,K*10).permute(0,2,1).contiguous() # [B, N, K, C] -> [B, N, K*C], C = 10 -> [B, F, N]
        FP = self.LFE(Pf)
        FQ = self.LFE(Qf)
        A = self.AML(FP, FQ)
        AN = self.IN(A.reshape(B,1,N,N)).squeeze(dim=1)
        M = self.SL(AN)
        R,t = self.TE(Ppc, Qpc, M)
        return M, R, t