# Various Losses used for training the network
import torch
import pytorch3d
# import kornia
import pytorch3d
import pytorch3d.loss
import numpy as np
import cv2
import torch
from pytorch3d.ops.knn import knn_points
from torch.nn.functional import mse_loss as mse


def velocity_loss(v_arr, d2v_dh2_arr, alpha=0.01):
    '''
    compute regularization term
    |[I-alpha*nabla^2] v|^2
    where nabla^2 is the diagonal Laplacian operator
    and I identity operator
    '''
    loss = 0
    T = 0
    
    for v, d2v_dh2 in zip(v_arr, d2v_dh2_arr):
        laplace_v = 0
        if alpha != 0:
            laplace_v = d2v_dh2.sum(-1)
        loss = loss + ( (v - alpha * laplace_v)**2 ).mean()
        T = T + 1
        
    return loss / T

def clipped_huber(x, y, max_val=1, beta=0.01):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = y-x
    diff = torch.where(mask, torch.zeros_like(diff), diff).abs()
    diff = torch.where(diff<=beta, (diff**2), 2*beta*diff.abs()-beta**2 )
    return diff.mean()

def clipped_mse(x, y, max_val=1, edge_lambda=None):
    x = x / max_val
    y = y / max_val
    mask = (x >= 1) & (y >= 1)
    diff = (x - y)**2
    diff = torch.where(mask, torch.zeros_like(diff), diff)
    loss = diff.mean()
    if edge_lambda is not None and edge_lambda != 0:
        edge_mse = mse(kornia.filters.spatial_gradient(x.clamp_max(1).permute(0,3,1,2)), 
                        kornia.filters.spatial_gradient(y.clamp_max(1).permute(0,3,1,2)))
        loss = loss + edge_lambda * edge_mse
    return loss

def clipped_mae(x, y, max_val=1):
    diff = (x - y).abs()
    img = (diff[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
    cv2.imwrite(f"./out/diff.png", img)
    return diff.mean()