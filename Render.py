import numpy as np
import torch
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PointLights, SoftPhongShader, DirectionalLights
)

from CookTorranceRendering import HardCookTorranceShader, SoftCookTorranceShader, SoftNormalShader, SoftXYZShader, SoftTextureShader
from CustomRendering import SoftCustomShader, HardCustomShader
from MerlRendering import HardMerlShader, SoftMerlShader, HardMerlMultiLightsShader
from utils import r2R

def render_mesh(mesh, modes, rotations, translations, image_size, blur_radius, faces_per_pixel, L0,
                device, background_colors=None, light_poses=None, materials=None, camera_settings=None, verts_radiance=None, multi_lights=None, ambient_net=None, sigma=1e-4, gamma=1e-4):

    # if isinstance(modes, str):
    #     return render_mesh(mesh, (modes,), rotations, translations, image_size, blur_radius, faces_per_pixel, 
    #             device, background_colors, light_poses, materials, camera_settings, verts_radiance, multi_lights, ambient_net, sigma, gamma)[0]

    if camera_settings is None:
        cameras = OpenGLPerspectiveCameras(device=device)
    else:
        cameras = PerspectiveCameras(device=device, **camera_settings)
    

    if background_colors is None:
        background_colors = [None] * len(modes)

    # rasterization
    raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius= min(np.log(1. / 1e-6 - 1.) * sigma, blur_radius / image_size * 2),  #np.log(1. / 1e-1 - 1.)*sigma, for mask
            faces_per_pixel=faces_per_pixel, 
            perspective_correct= False,
            max_faces_per_bin=50000
        )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    bgc = (0,0,0) # background color is black
    NeRF_bgc = None
    blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=bgc)
    
    t = (-torch.inverse(rotations[0]) @ translations[0])[None] # translation in camera space
    if modes == 'image_ct':
        shader = SoftCookTorranceShader(device=device, cameras=cameras, blend_params=blend_params, NeRF_bgc=NeRF_bgc)      
        lights = PointLights(device=device, location=translations[0][None],diffuse_color=torch.tensor([[L0,L0,L0]], device=device))
        fragments = rasterizer(mesh, R=rotations, T=t)
        images = shader(fragments, mesh, lights=lights)
    elif modes == 'silhouette':
        shader = SoftSilhouetteShader(blend_params=blend_params)       
        fragments = rasterizer(mesh, R=rotations, T=t)
        images = shader(fragments, mesh)
        images= images[0,...,3:4]

    return images

 

