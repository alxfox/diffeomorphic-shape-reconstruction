import numpy as np
import torch
import pytorch3d
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.io import load_ply
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
import imageio
from Render import render_mesh
from utils import r2R, dotty, save_images, pytorch_camera, Meshes
import cv2
import os
import pickle
from os.path import isfile, join




def create(viewpoints, validation = False, name = None):
    device = torch.device("cuda:0")
    viewpoints = np.load(join('data',viewpoints))
    
    verts, faces = load_ply("data/cubes.ply")
    verts_rgb = (torch.ones_like(verts)*torch.tensor([0,1,0]))[None]  # color the cubes green
    cube = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], vert_textures=verts_rgb.to(device))

    verts, faces = load_ply("data/mesh.ply")
    verts_rgb = torch.ones_like(verts)[None] # color the bunny white
    bunny = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], vert_textures=verts_rgb.to(device))

    mesh = join_meshes_as_scene([bunny, cube], include_textures=True)
    
    params = dotty({
    'device': torch.device("cuda"),
    'rendering':
    {
        'rgb': 
        {
            'image_size': 100, #######
            'blur_radius': 0.0, #######
            'faces_per_pixel': 4, #######
            'max_intensity': 0.15, #######   read_ing:0.09, budd_ha: 0.15, pot_2: 0.15, co_w: 0.15, bea_r:  0.2
            'sigma': 1e-4, #######
            'gamma': 1e-4, #######
            'L0': 10 # set this to none when creating dataset of cubes only
        },
        'silhouette':
        {
            'image_size': 100, #######
            'blur_radius': 0.1,#np.log(1. / 1e-4 - 1.) * 1e-4
            'faces_per_pixel': 100,
            'sigma': 1e-4,
            'gamma': 1e-4,
            'L0':None
        }
    }})

    n_images=30
    R = torch.zeros((n_images, 3, 3), device=device)
    T = torch.zeros((n_images, 3), device=device)
    for i in range(n_images): # read camera extrinsics
        old_rotation = torch.from_numpy(viewpoints["R_"+str(i)]).float().to(device)
        # fix rotation
        extra_rotation = torch.tensor([[-1.0,0,0],[0,-1.0,0],[0,0,1.0]],dtype=old_rotation.dtype).to(device)
        new_rotation = old_rotation@extra_rotation
        R[i] = new_rotation
        T[i] = torch.from_numpy(viewpoints["T_"+str(i)]).float().to(device)
    
    
    camera_settings = pytorch_camera(params['rendering.rgb.image_size'], torch.from_numpy(viewpoints["K"]).float().to(device))

    images = []
    silhouettes =[]
    max_val = 0
    if(validation): 
        f = open('store.pckl', 'rb')
        params['rendering.rgb.L0'] = pickle.load(f).item()
        f.close()

    for i in range(n_images):
        prd_image = render_mesh(mesh, 
                                modes='image_ct', #######
                                rotations=R[i:i+1], 
                                translations=T[i:i+1], 
                                image_size=params['rendering.rgb.image_size'], 
                                blur_radius=params['rendering.rgb.blur_radius'], 
                                faces_per_pixel=params['rendering.rgb.faces_per_pixel'], 
                                L0=params['rendering.rgb.L0'],
                                device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings,
                                sigma=params['rendering.rgb.sigma'], gamma=params['rendering.rgb.gamma'], name='dataset')
        prd_image = prd_image[...,:3]
        images.append(prd_image)
        max_val = max(max_val,prd_image.max())
        sh_image = render_mesh(mesh, 
                        modes='silhouette', #######
                        rotations=R[i:i+1], 
                        translations=T[i:i+1], 
                        image_size=params['rendering.silhouette.image_size'], 
                        blur_radius=params['rendering.silhouette.blur_radius'], 
                        faces_per_pixel=params['rendering.silhouette.faces_per_pixel'], 
                        L0=params['rendering.silhouette.L0'],
                        device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings,
                        sigma=params['rendering.silhouette.sigma'], gamma=params['rendering.silhouette.gamma'])
        silhouettes.append(sh_image)

    # Comment out these lines when creating dataset of cubes only
    if(not validation):
        new_L0 = params['rendering.rgb.L0']/max_val
        f = open('store.pckl', 'wb')
        pickle.dump(new_L0, f)
        f.close()

    path = './data/dataset'
    if os.path.exists(path)== False:
        os.mkdir(path)
    
    for i in range(n_images):
    # images are saved to out as a png    
        img = (images[i]/max_val*((256**2)-1)).cpu().numpy().astype(np.uint16)[0]
        imgsh = (silhouettes[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)
        #writer.append_data(img)
        if (validation):
            cv2.imwrite(f"./data/dataset/cubesmesh_render_{name}{i:02}.png", img)
        else:
            cv2.imwrite(f"./data/dataset/cubesmesh_render_{i:02}.png", img)
            cv2.imwrite(f"./data/dataset/cubesmesh_mask_{i:02}.png", imgsh)
        
        
    
    
if __name__ == '__main__':
     
    create(viewpoints = 'cameras1.npz')
    create(viewpoints = 'cameras_behind.npz', validation = True , name = 'behind')
    create(viewpoints = 'cameras_above.npz', validation = True , name = 'above')
    create(viewpoints = 'cameras_below.npz', validation = True , name = 'below')
    
      
    