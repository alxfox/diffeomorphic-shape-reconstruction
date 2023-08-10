import numpy as np
import torch
import cv2
import os
import pickle
from os.path import join
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.io import load_ply
from Render import render_mesh
from utils import dotty, pytorch_camera, Meshes
from tqdm import tqdm

def create(viewpoints, dataset_name, view_name, validation = False, cubes_color=[1,1,1]):
    print("creating dataset:", dataset_name, "|", "view:", view_name)
    device = torch.device("cuda:0")
    viewpoints = np.load(join('data', viewpoints))
    if(dataset_name=="rotation"):
        verts, faces = load_ply("data/rotation_cubes.ply")
    elif(dataset_name=="normal"):
        verts, faces = load_ply("data/cubes.ply")
    verts_rgb = (torch.ones_like(verts) * torch.tensor(cubes_color))[None]  # color the cubes
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
            'image_size': 100, 
            'blur_radius': 0.0, 
            'faces_per_pixel': 4, 
            'max_intensity': 0.15,
            'sigma': 1e-4, 
            'gamma': 1e-4, 
            'L0': 10
        },
        'silhouette':
        {
            'image_size': 100, 
            'blur_radius': 0.1,
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
    
    # render the cubes with the bunny:
    images = []
    silhouettes =[]
    max_val = 0

    if(validation): 
        f = open('store.pckl', 'rb')
        params['rendering.rgb.L0'] = pickle.load(f).item()
        f.close()

    for i in tqdm(range(n_images)):
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

    path = './data/dataset/' + dataset_name
    if os.path.exists(path + "/full")== False:
        os.makedirs(path + "/full")
    if(not validation):
        params['rendering.rgb.L0'] = params['rendering.rgb.L0']/max_val
        f = open(path+'/store.pckl', 'wb')
        pickle.dump(params['rendering.rgb.L0'], f)
        f.close()

    for i in range(n_images):
    # images are saved to out as a png    
        img = (images[i]/max_val*((256**2)-1)).cpu().numpy().astype(np.uint16)[0]
        imgsh = (silhouettes[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)
        #writer.append_data(img)
        if (validation):
            cv2.imwrite(f"{path}/full/render_{view_name}{i:02}.png", img)
        else:
            cv2.imwrite(f"{path}/full/render_{view_name if view_name!=None else ''}_{i:02}.png", img)
            cv2.imwrite(f"{path}/full/mask_{view_name if view_name!=None else ''}_{i:02}.png", imgsh)
    
    # render only the cubes:
    images = []
    silhouettes = []
    for i in tqdm(range(n_images)):
        prd_image = render_mesh(cube, 
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

        sh_image = render_mesh(cube, 
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
    if os.path.exists(path + "/cubes")== False:
        os.makedirs(path + "/cubes")
    for i in range(n_images):
    # images are saved to out as a png    
        img = (images[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)[0]
        imgsh = (silhouettes[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)
        if (validation):
            cv2.imwrite(f"{path}/cubes/render_{view_name}_{i:02}.png", img)
        else:
            cv2.imwrite(f"{path}/cubes/render_{view_name}_{i:02}.png", img)
            cv2.imwrite(f"{path}/cubes/mask_{view_name}_{i:02}.png", imgsh)

    # render only the bunny:
    images = []
    silhouettes = []
    for i in tqdm(range(n_images)):
        prd_image = render_mesh(bunny, 
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

        sh_image = render_mesh(bunny, 
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
    if os.path.exists(path + "/bunny")== False:
        os.makedirs(path + "/bunny")
    for i in range(n_images):
    # images are saved to out as a png    
        img = (images[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)[0]
        imgsh = (silhouettes[i]*((256**2)-1)).cpu().numpy().astype(np.uint16)
        if (validation):
            cv2.imwrite(f"{path}/bunny/render_{view_name}_{i:02}.png", img)
        else:
            cv2.imwrite(f"{path}/bunny/render_{view_name}_{i:02}.png", img)
            cv2.imwrite(f"{path}/bunny/mask_{view_name}_{i:02}.png", imgsh)
    
    
if __name__ == '__main__':
    print("starting dataset creation...")
    # cubes_color = [1,1,1] # white
    cubes_color = [0,1,0] # green
    create(viewpoints = 'cameras1.npz', dataset_name="normal", validation = False, view_name = 'front', cubes_color=cubes_color)
    create(viewpoints = 'cameras_behind.npz', dataset_name="normal", validation = True, view_name = 'behind', cubes_color=cubes_color)
    create(viewpoints = 'cameras_above.npz', dataset_name="normal", validation = True, view_name = 'above', cubes_color=cubes_color)
    create(viewpoints = 'cameras_below.npz', dataset_name="normal", validation = True, view_name = 'below', cubes_color=cubes_color)

    create(viewpoints = 'cameras_behind.npz', dataset_name="rotation", validation = False, view_name = 'behind', cubes_color=cubes_color)
    create(viewpoints = 'cameras1.npz', dataset_name="rotation", validation = True, view_name = 'front', cubes_color=cubes_color)
    create(viewpoints = 'cameras_above.npz', dataset_name="rotation", validation = True, view_name = 'above', cubes_color=cubes_color)
    create(viewpoints = 'cameras_below.npz', dataset_name="rotation", validation = True, view_name = 'below', cubes_color=cubes_color)