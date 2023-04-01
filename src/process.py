import json
import datetime
from dataset import dataset
from utils.metrics import R_error, t_error

class PCRProcess():
    def __init__(self,h5fs_list,):
        self.current=0
        self.data=dataset(h5fs_list=h5fs_list)
        self.total=self.data.__len__()
        self.cur_src_points=None
        self.cur_src_normals=None
        self.cur_src_knns=None
        self.cur_tgt_points=None
        self.cur_tgt_normals=None
        self.cur_tgt_knns=None
        self.cur_gt_R=None
        self.cur_gt_t=None

    def registration(self, PCRModel, methodName):
        if methodName=='DGM-Net':
            return PCRModel(self.cur_src_points, self.cur_src_normals, self.cur_src_knns, self.cur_tgt_points, self.cur_tgt_normals,self.cur_tgt_knns)
        return None

    def move(self):
        if self.current<self.total:
            self.cur_src_points,\
            self.cur_src_normals,\
            self.cur_src_knns,\
            self.cur_tgt_points,\
            self.cur_tgt_normals,\
            self.cur_tgt_knns,\
            self.cur_gt_R,\
            self.cur_gt_t=self.data.__getitem__(self.current)
            self.current+=1
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
        
    
    