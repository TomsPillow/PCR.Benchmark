import torch
import time
from baselines.dgmnet.model.dgmnet import DGMNet
from baselines.pcrnet.model.pcrnet import IterativePCRNet
from baselines.dgm.model.dgm import DGM
from baselines.icp.model.icp import ICP
from baselines.fgr.model.fgr import FGR
from baselines.gmm.model.gmm import GMM

model=None
model_type=None
model_checkpoint=None
def DGMNetAPI(checkpoint, src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns):
    global model_type
    global model
    global model_checkpoint
    src_points=torch.as_tensor(src_points).unsqueeze(0)
    src_normals=torch.as_tensor(src_normals).unsqueeze(0)
    src_knns=torch.as_tensor(src_knns).unsqueeze(0).long()
    tgt_points=torch.as_tensor(tgt_points).unsqueeze(0)
    tgt_normals=torch.as_tensor(tgt_normals).unsqueeze(0)
    tgt_knns=torch.as_tensor(tgt_knns).unsqueeze(0).long()

    if model_type==None or model_type!='dgmnet':
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=DGMNet().to(device)
        model_type='dgmnet'
    if model_checkpoint==None or model_checkpoint!=checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model_checkpoint=checkpoint

    start=time.time()
    _, R, t = model(src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns)
    end=time.time()
    time_cost=end-start
    return R.cpu().detach().numpy()[0],t.detach().cpu().numpy()[0], time_cost

def PCRNetAPI(checkpoint, src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns):
    global model_type
    global model
    global model_checkpoint
    src_points=torch.as_tensor(src_points.transpose(1,0)).unsqueeze(0)
    tgt_points=torch.as_tensor(tgt_points.transpose(1,0)).unsqueeze(0)
    
    if model_type==None or model_type!='pcrnet':
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=IterativePCRNet().to(device)
        model_type='pcrnet'
    if model_checkpoint==None or model_checkpoint!=checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model_checkpoint=checkpoint

    start=time.time()
    R, t = model(src_points, tgt_points)
    end=time.time()
    time_cost=end-start
    return R.cpu().detach().numpy()[0],t.detach().cpu().numpy()[0], time_cost

def DGMAPI(src_points, src_normals, tgt_points, tgt_normals):
    global model_type
    global model
    if model_type==None or model_type!='dgm':
        model=DGM()
        model_type='dgm'
    start=time.time()
    R, t, _, _ = model(src_points, src_normals, tgt_points, tgt_normals)
    end=time.time()
    time_cost=end-start
    return R, t, time_cost

def ICPAPI(src_points, src_normals, tgt_points, tgt_normals):
    global model_type
    global model
    if model_type==None or model_type!='icp':
        model=ICP()
        model_type='icp'
    start=time.time()
    R, t = model(src_points, src_normals, tgt_points, tgt_normals)
    end=time.time()
    time_cost=end-start
    return R, t, time_cost

def FGRAPI(src_points, src_normals, tgt_points, tgt_normals):
    global model_type
    global model
    if model_type==None or model_type !='fgr':
        model=FGR()
        model_type='fgr'
    start=time.time()
    R, t = model(src=src_points, tgt=tgt_points)
    end=time.time()
    time_cost=end-start
    return R, t, time_cost

def GMMAPI(src_points, src_normals, tgt_points, tgt_normals):
    global model_type
    global model
    if model_type==None or model_type !='gmm':
        model=GMM()
        model_type='gmm'
    start=time.time()
    R, t = model(src=src_points, tgt=tgt_points)
    end=time.time()
    time_cost=end-start
    return R, t, time_cost
