import json
import datetime
from dataset import dataset
from utils.metrics import R_error, t_error
from utils.transform import transform

DP_PCRMethods=['DGMNet','PCRNet']
NORM_PCRMethods=['DGM','ICP','FGR','GMM','D3Feat']

class PCRProcess():
    def __init__(self,h5fs_list,GaussNoise=False,sigma=0.1,bias=0.05):
        self.current=-1
        self.data=dataset(h5fs_list=h5fs_list,noise=GaussNoise,sigma=sigma,clip=bias)
        self.total=self.data.__len__()
        self.cur_src_points=None
        self.cur_src_normals=None
        self.cur_src_knns=None
        self.cur_tgt_points=None
        self.cur_tgt_normals=None
        self.cur_tgt_knns=None
        self.cur_gt_R=None
        self.cur_gt_t=None

    def registration(self, PCRModel, methodName, checkpoint=None):
        if methodName in DP_PCRMethods:
            return PCRModel(checkpoint, self.cur_src_points, self.cur_src_normals, self.cur_src_knns, self.cur_tgt_points, self.cur_tgt_normals,self.cur_tgt_knns)
        elif methodName in NORM_PCRMethods:
            return PCRModel(self.cur_src_points,self.cur_src_normals,self.cur_tgt_points,self.cur_tgt_normals)
        return None

    def move(self):
        if self.current<self.total:
            self.current+=1
            self.cur_src_points,\
            self.cur_src_normals,\
            self.cur_src_knns,\
            self.cur_tgt_points,\
            self.cur_tgt_normals,\
            self.cur_tgt_knns,\
            self.cur_gt_R,\
            self.cur_gt_t=self.data.__getitem__(self.current)
            
            return self.cur_src_points, self.cur_tgt_points, self.cur_gt_R, self.cur_gt_t, self.current, self.total
        
    def back(self):
        if self.current>0:
            self.current-=1
            self.cur_src_points,\
            self.cur_src_normals,\
            self.cur_src_knns,\
            self.cur_tgt_points,\
            self.cur_tgt_normals,\
            self.cur_tgt_knns,\
            self.cur_gt_R,\
            self.cur_gt_t=self.data.__getitem__(self.current)
            return self.cur_src_points, self.cur_tgt_points, self.cur_gt_R, self.cur_gt_t, self.current, self.total
        
    def metrics(self, est_R, est_t):
        (R_MSE,R_MAE),R_isotropic=R_error(est_R, self.cur_gt_R)
        (t_MSE,t_MAE),t_isotropic=t_error(est_t, self.cur_gt_t)
        return R_MSE,R_MAE,R_isotropic,t_MSE,t_MAE,t_isotropic
        
    def getCurrent(self):
        return self.current

    def getTotal(self):
        return self.total
    
    def getProgress(self):
        return self.current / self.total
        
    def transform(self, R, t):
        return transform(self.cur_src_points.transpose(1,0), R, t).transpose(1,0)
    