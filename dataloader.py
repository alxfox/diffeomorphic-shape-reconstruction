# Dataloading for training and validation datasets
import numpy as np
import torch
import scipy
import scipy.io
import cv2
import trimesh


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

def load_dataset(path_str, device, n_images=30, dataset_name="normal", use_background=False, provide_background=False, max_faces_no=None):
    if(dataset_name=="normal"):
        viewpoints_name = "front"
        cameras = np.load(f'{path_str}/cameras1.npz')
    elif(dataset_name=="rotation"):
        viewpoints_name = "behind"
        cameras = np.load(f'{path_str}/cameras_behind.npz')
    else:
        return # dataset doesn't exist
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
                    cv2.imread(f'{path_str}/dataset/{dataset_name}/{"full" if use_background else "bunny"}/render_{viewpoints_name}_{j:02}.png', -1)[...,::-1].astype(np.float32) \
                    for j in range(n_images)], axis=0)).to(device)[...,:3]/(256**2-1)
    # Silhouettes are not used anymore
    masks = torch.zeros_like(images)
    # masks = torch.from_numpy(np.stack([\
    #                         (cv2.imread(f'{path_str}/dataset/cubesmesh_mask_{j:02}.png', -1)).astype(np.float32)\
    #                         for j in range(n_images)], axis=0)).to(device)/(256**2-1)
    if(provide_background):
        background = torch.from_numpy(np.stack([\
                    cv2.imread(f'{path_str}/dataset/{dataset_name}/cubes/render_{viewpoints_name}_{j:02}.png', -1)[...,::-1].astype(np.float32) \
                    for j in range(n_images)], axis=0)).to(device)[...,:3]/(256**2-1)
    else:
        background = torch.zeros_like(images) # black background

    return images, masks, background, R, T, K, None

def load_val_dataset(path_str, device, n_images=30, dataset_name="normal", viewpoints_name=None, use_background=False, max_faces_no=None):
    
    cameras = np.load(f'{path_str}/cameras_{viewpoints_name}.npz')

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
                     cv2.imread(f'{path_str}/dataset/{dataset_name}/cubes/render_{viewpoints_name}_{j:02}.png', -1)[...,::-1].astype(np.float32) \
                for j in range(n_images)], axis=0)).to(device)[...,:3]/(256**2-1)
   
    masks = torch.zeros_like(images)
    #       torch.from_numpy(np.stack([\
    #                  cv2.imread(f'{path_str}/dataset/cubes/mask_{j:02}.png', -1).astype(np.float32)\
    #             for j in range(n_images)], axis=0)).to(device)/(256**2-1)
    if(use_background):
        background = torch.from_numpy(np.stack([\
                    cv2.imread(f'{path_str}/dataset/{dataset_name}/cubes/render_{viewpoints_name}_{j:02}.png', -1)[...,::-1].astype(np.float32) \
                for j in range(n_images)], axis=0)).to(device)[...,:3]/(256**2-1)
    else:
        background = torch.zeros_like(images) # black background

    return images, masks, background, R, T, K, None