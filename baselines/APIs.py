import torch
import time
from baselines.dgmnet.model.dgmnet import DGMNet

model=None
model_type=None

def DGMNetAPI(src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns):
    global model_type
    global model
    src_points=torch.as_tensor(src_points).unsqueeze(0)
    src_normals=torch.as_tensor(src_normals).unsqueeze(0)
    src_knns=torch.as_tensor(src_knns).unsqueeze(0).long()
    tgt_points=torch.as_tensor(tgt_points).unsqueeze(0)
    tgt_normals=torch.as_tensor(tgt_normals).unsqueeze(0)
    tgt_knns=torch.as_tensor(tgt_knns).unsqueeze(0).long()

    if model_type==None or model_type!='dgmnet':
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=DGMNet().to(device)
        model.load_state_dict(torch.load('./baselines/dgmnet/checkpoints/epoch_240_fdgmnet_params.pth',map_location=device))

    start=time.time()
    _, R, t = model(src_points, src_normals, src_knns, tgt_points, tgt_normals, tgt_knns)
    end=time.time()
    time_cost=end-start
    return R.cpu().detach().numpy()[0],t.detach().cpu().numpy()[0], time_cost


def DGMAPI():
    pass

def ICPAPI():
    pass