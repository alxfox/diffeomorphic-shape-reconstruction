# Differentiable Renderer
import numpy as np
import torch
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, PerspectiveCameras, RasterizationSettings, 
    MeshRasterizer, BlendParams, SoftSilhouetteShader, PointLights
)

from CookTorranceRendering import SoftCookTorranceShader


def render_mesh(mesh, modes, rotations, translations, image_size, blur_radius, faces_per_pixel, L0,
                device, background_colors=None, NeRF_bgc=None, light_poses=None, materials=None, camera_settings=None, verts_radiance=None, multi_lights=None, ambient_net=None, sigma=1e-4, gamma=1e-4, name=None):

    if camera_settings is None:
        cameras = OpenGLPerspectiveCameras(device=device)
    else:
        cameras = PerspectiveCameras(device=device, **camera_settings)
    

    if background_colors is None:
        background_colors = [None] * len(modes)

    # rasterization
    raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius= min(np.log(1. / 1e-6 - 1.) * sigma, blur_radius / image_size * 2), 
            faces_per_pixel=faces_per_pixel, 
            perspective_correct= False,
            max_faces_per_bin=50000
        )
    
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    bgc = (0,0,0) # background color is black
    blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
    
    t = (-torch.inverse(rotations[0]) @ translations[0])[None] # translation in camera space

    if modes == 'image_ct':
        shader = SoftCookTorranceShader(device=device, cameras=cameras, blend_params=blend_params, NeRF_bgc=NeRF_bgc, name=name)      
        lights = PointLights(device=device, location=translations[0][None],diffuse_color=torch.tensor([[L0,L0,L0]], device=device))
        fragments = rasterizer(mesh, R=rotations, T=t)
        images = shader(fragments, mesh, lights=lights)
    elif modes == 'silhouette':
        shader = SoftSilhouetteShader(blend_params=blend_params)       
        fragments = rasterizer(mesh, R=rotations, T=t)
        images = shader(fragments, mesh)
        #return alpha channel
        images= images[0,...,3:4]

    return images