from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import scipy
import scipy.io
import cv2
import trimesh
import sys
from utils import dotty, P_matrix_to_rot_trans_vectors, pytorch_camera
from Render import render_mesh
from Meshes import Meshes
from utils import save_images
import pytorch3d
from Loss import chamfer_3d

def make_torch_tensor(*args):
    return tuple(torch.tensor(a).cuda() for a in args)

def load_mesh(path_str, name_str):
    mesh_path = f'{path_str}/mesh_Gt.ply'

    mesh = trimesh.load(mesh_path)
    verts, faces = make_torch_tensor(mesh.vertices.astype(np.float32), mesh.faces)

    shift = verts.mean(0)
    scale = (verts - shift).abs().max()

    transf = torch.eye(4).cuda()
    transf[:3,:3] = torch.eye(3).cuda() * scale
    transf[:3,3] = shift

    return (verts - shift) / scale, faces, transf

def load_dataset(path_str, device, n_images=30, max_faces_no=None):
    # lights_ints = np.loadtxt(f'{path_str}/view_{view_id:02}/light_intensities.txt').astype(np.float32)
    # light_dirs = np.loadtxt(f'{path_str}/view_{view_id:02}/light_directions.txt')#@np.diag([-1,1,1])
    # light_dirs = light_dirs.astype(np.float32)
    
    cameras = np.load(f'{path_str}/cameras1.npz')

    R = torch.zeros((n_images, 3, 3), device=device)
    T = torch.zeros((n_images, 3), device=device)

    for i in range(n_images): # read camera extrinsics
        old_rotation = torch.from_numpy(cameras["R_"+str(i)]).float().to(device)
        # fix rotation
        extra_rotation = torch.tensor([[-1.0,0,0],[0,-1.0,0],[0,0,1.0]], dtype=old_rotation.dtype).to(device)
        new_rotation = old_rotation @ extra_rotation
        R[i] = new_rotation
        T[i] = torch.from_numpy(cameras["T_"+str(i)]).float()
    K = torch.from_numpy(cameras["K"]).float().to(device)
    images = torch.from_numpy(np.stack([\
                     cv2.imread(f'{path_str}/dataset/render_{j:02}.png', -1)[...,::-1].astype(np.float32) \
                for j in range(n_images)], axis=0)).to(device)[...,:3]/(256**2-1)
    # imgs = imgs / lights_ints.reshape(lights_ints.shape[0], 1, 1, lights_ints.shape[1]) / 65535

    masks = torch.from_numpy(np.stack([\
                            (cv2.imread(f'{path_str}/dataset/mask_{j:02}.png', -1)).astype(np.float32)\
                             for j in range(n_images)], axis=0)).to(device)/(256**2-1)
    # normal = scipy.io.loadmat(f'{path_str}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32)

    # colocated_mask = (light_dirs[...,-1] > 0.65)
    return images, masks, R, T, K, None #, transf
    # return make_torch_tensor(imgs.astype(np.float32)[colocated_mask], 
    #     mask.astype(np.float32), (light_dirs[colocated_mask]@np.diag([1,-1,-1])@R).astype(np.float32), view_dirs[colocated_mask], normal, K, P)

@torch.no_grad()
def diligent_eval(verts, faces, gt_verts, gt_faces, path, transf, K, Ps, masks=None, mode='normal', transform_verts=False, normals=None):
    
    r, t = P_matrix_to_rot_trans_vectors(Ps)
    camera_settings = pytorch_camera(512, K)
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]

        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)

        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
            if normals is not None:
                normals = normals @ torch.inverse(R).transpose(-1,-2)
            

    estimates = render_mesh(Meshes(verts=[verts], faces=[faces], verts_normals=([normals] if normals is not None else None)), 
                        modes=mode, #######
                        rotations=r, 
                        translations=t, 
                        image_size=512, 
                        blur_radius=0.00, 
                        faces_per_pixel=4, 
                        device=verts.device, background_colors=None, camera_settings=camera_settings,
                        sigma=1e-4, gamma=1e-4)

    ground_truths = render_mesh(Meshes(verts=[gt_verts], faces=[gt_faces]), 
                        modes=mode, #######
                        rotations=r, 
                        translations=t, 
                        image_size=512, 
                        blur_radius=0, 
                        faces_per_pixel=8, 
                        device=verts.device, background_colors=None, camera_settings=camera_settings,
                        sigma=1e-4, gamma=1e-4)

    if mode == 'depth':
        estimates = estimates.unsqueeze(dim=-1)
        ground_truths = ground_truths.unsqueeze(dim=-1)
    elif mode == 'normal':
        shape = estimates.shape

        estimates = torch.nn.functional.normalize((estimates.reshape(shape[0], -1, 3)@Ps[:,:3,:3].transpose(-1,-2)).reshape(shape), dim=-1)
        estimates = estimates * torch.tensor([1,-1,-1], dtype=estimates.dtype, device=estimates.device)

        ground_truths = np.stack([scipy.io.loadmat(f'{path}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32) for view_id in range(1,21)], 0)
        ground_truths = torch.from_numpy(ground_truths).cuda()[:,:,50:-50]
        ground_truths = torch.nn.functional.normalize(ground_truths, dim=-1)

        save_images((estimates+1)/2, 'est', 1)
        save_images((ground_truths+1)/2, 'gt', 1)

    fg_masks = (torch.norm(estimates, dim=-1) > 0) & (torch.norm(ground_truths, dim=-1) > 0)

    if masks is not None:
        fg_masks = fg_masks & masks
    
    if mode == 'normal':
        error = torch.acos((estimates * ground_truths).sum(-1).clamp(-1,1)) * 180 / np.pi
    elif mode == 'depth':
        error = (estimates - ground_truths).abs() * scale

    error[~fg_masks] = 0

    return error, fg_masks


def eval_normal(path_str, name_str, method):
    path = f'{path_str}/mvpmsData/{name_str}PNG'
    ground_truths = np.stack([scipy.io.loadmat(f'{path}/view_{view_id:02}/Normal_gt.mat')['Normal_gt'].astype(np.float32) for view_id in range(1,21)], 0)
    
    if method == 'li':
        path = f'{path_str}/estNormal/{name_str}PNG_Normal_TIP19Li_view'
    elif method == 'park':
        path = f'{path_str}/estNormal/{name_str}PNG_Normal_PAMI16Park_view'
    
    estimates = np.stack([scipy.io.loadmat(f'{path}{view_id}.mat')['Normal_est'].astype(np.float32) for view_id in range(1,21)], 0)

    fg_masks = (np.linalg.norm(estimates, axis=-1) > 0) & (np.linalg.norm(ground_truths, axis=-1) > 0)

    error = np.arccos((estimates * ground_truths).sum(-1).clip(-1,1)) * 180 / np.pi
    error[~fg_masks] = 0

    return error, fg_masks

@torch.no_grad()
def eval_chamfer(verts, faces, gt_verts, gt_faces, transf, transform_verts=False, n_points=1000000, z_thresh=0):
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]
        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)
        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)        

    mesh_est = Meshes(verts=[verts], faces=[faces])
    mesh_gt = Meshes(verts=[gt_verts], faces=[gt_faces])

    pcd_est = pytorch3d.ops.sample_points_from_meshes(mesh_est, n_points).reshape((-1,n_points,3))
    pcd_gt = pytorch3d.ops.sample_points_from_meshes(mesh_gt, n_points).reshape((-1,n_points,3))

    return chamfer_3d(pcd_est, pcd_gt, z_thresh) * scale

from Loss import mesh2mesh_chamfer
@torch.no_grad()
def eval_mesh_to_mesh(verts, faces, gt_verts, gt_faces, transf, transform_verts=False, n_points=1000000):
    scale = 1

    if transf is not None:
        transf = transf / transf[...,-1:,-1:]
        R = transf[...,:3,:3]
        T = transf[...,:3, 3]

        gt_verts = (gt_verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)
        scale = float(torch.det(R.reshape(-1,3,3)[0])) ** (1/3.0)

        if transform_verts:
            verts = (verts - T.unsqueeze(-2)) @ torch.inverse(R).transpose(-1,-2)

    mesh_est = Meshes(verts=[verts], faces=[faces])
    mesh_gt = Meshes(verts=[gt_verts], faces=[gt_faces])

    return mesh2mesh_chamfer(mesh_est, mesh_gt, n_points) * scale