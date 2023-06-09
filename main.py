import numpy as np
import torch
import pytorch3d
from pytorch3d.io import load_ply
import matplotlib.pyplot as plt
import imageio
from Render import render_mesh
from utils import r2R, dotty, save_images, pytorch_camera, Meshes
import cv2


if __name__ == '__main__':
    device = torch.device("cuda:0")
    verts, faces = load_ply("mesh_noNormals.ply")
    viewpoints = np.load('cameras.npz')
   
    verts_rgb = torch.ones_like(verts)[None]  # color the mesh white
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], vert_textures=verts_rgb.to(device))

    

    
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
            'L0': 10
        }
    }})

    n_images=29
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

    writer = imageio.get_writer("./out/render.gif", mode='I', duration=0.3)
    images = None
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
                                sigma=params['rendering.rgb.sigma'], gamma=params['rendering.rgb.gamma'])
        prd_image = prd_image[...,:3]
        if(images == None):
            images = prd_image.unsqueeze(0)
        else:
            images = torch.cat((images, prd_image.unsqueeze(0)), dim=0)
        # images are saved to out as a png 
        new_L0 = params['rendering.rgb.L0']/prd_image.max()
        img = (prd_image/prd_image.max()*255).cpu().type(torch.uint8).numpy()[0]
        writer.append_data(img)
        cv2.imwrite("./out/render"+"_"+str(i)+".png", img)

    
        
    