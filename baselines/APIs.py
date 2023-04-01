import torch
import time
from baselines.dgmnet.model import dgmnet

model=None
model_type=None
checkpoint_loaded=False

def DGMNetAPI(src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns):
    src_points=torch.as_tensor(src_points)
    src_normals=torch.as_tensor(src_normals)
    src_knns=torch.as_tensor(src_knns)
    tgt_points=torch.as_tensor(tgt_points)
    tgt_normals=torch.as_tensor(tgt_normals)
    tgt_knns=torch.as_tensor(tgt_knns)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if modelType==None or model_type!='dgmnet':
        model=DGMNet(K=args.K).to(device)
    if not checkpoint_loaded:
        model.load_state_dict(torch.load('./baselines/dgmnet/checkpoints/epoch_240_fdgmnet_params.pth')).to(device)
        checkpoint_loaded=True
    start=time.time()
    _, R, t = model(src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns)
    end=time.time()
    time_cost=end-start
    return R, t, time_cost


def DGMAPI():
    pass

def ICPAPI():
    pass