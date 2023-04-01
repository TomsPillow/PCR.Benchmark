import numpy as np
import torch
from scipy.spatial.transform import Rotation as rot

def R_error(R, gt_R):
    return R_anisotropic_error(R,gt_R),R_isotropic_error(R,gt_R)

def t_error(t, gt_t):
    return t_anisotropic_error(t,gt_t),t_isotropic_error(t,gt_t)

def R_anisotropic_error(R, gt_R):
    euler = rot.from_matrix(R).as_euler(seq='xyz',degrees=True)
    gt_euler = rot.from_matrix(gt_R).as_euler(seq='xyz',degrees=True)
    diff = euler-gt_euler
    R_mse = np.sum(diff**2)
    R_mae = np.sum(np.abs(diff))
    return R_mse, R_mae

def t_anisotropic_error(t, gt_t):
    t_mse = np.sum((t-gt_t)**2)
    t_mae = np.sum(np.abs(t-gt_t))
    return t_mse,t_mae

def R_isotropic_error(R,gt_R):
    return np.arccos((np.trace(np.matmul(gt_R.T,R))-1)/2) / np.pi * 180

def t_isotropic_error(t,gt_t):
    return np.sqrt(np.sum((gt_t-t)**2))