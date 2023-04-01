import json
import datetime
from dataset import dataset

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

    def registration(self, PCRModel):
        PCRModel()
        return

    def move(self):
        if self.current<self.total:
            self.cur_src_points,
            self.cur_src_normals,
            self.cur_src_knns,
            self.cur_tgt_points
            self.cur_tgt_normals,
            self.cur_tgt_knns,gt_R,gt_t=self.data.__getitem__(self.current)
            self.current+=1
            return self.cur_src_points, self.cur_tgt_points, gt_R, gt_t, self.current, self.total
        
    def getCurrent(self):
        return self.current

    def getTotal(self):
        return self.total
    
    def getProgress(self):
        return self.current / self.total
        
    
    