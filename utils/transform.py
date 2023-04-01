import numpy as np

def transform(pc, R, t):
    '''
    pc: 3 x N
    R: 3 x 3
    t: 3 x 1
    '''
    return np.matmul(R, pc,dtype=np.float32) + t.reshape(3,1)

def rotation(vecs, R):
    return np.matmul(R, vecs,dtype=np.float32)