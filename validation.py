import numpy as np
import torch
import pytorch3d
from pytorch3d.io import load_ply
from pytorch3d.io import load_obj
from Loss import velocity_loss, clipped_mae, chamfer_3d
import matplotlib.pyplot as plt
import imageio
from Render import render_mesh
from utils import r2R, dotty, save_images, pytorch_camera, Meshes
import cv2
from utils import manual_seed, rand_ico_sphere, save_models, load_models
import os
from os.path import isfile, join
from Model import MLP, PositionEncoding, ResNet, Sequential, ShapeNet, BRDFNet
import pickle
import yaml
from torch.nn.functional import mse_loss as mse
from dataloader import load_val_dataset,load_dataset


if __name__ == '__main__':
    device = torch.device("cuda:0")
    file = 'experiment_e852184d-ed34-438e-aa06-549f2a2ebe99'
    conf = join('./out',file,'config.yaml')
    
    config = yaml.safe_load(open(conf))
    #verts, faces = load_ply("data/mesh.ply")
    verts, faces, aux = load_obj(join('./out',file,"checkpoint.obj"))
      
    verts_rgb = torch.ones_like(verts)[None]  # color the mesh white
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], vert_textures=verts_rgb.to(device)).to(device)
    #mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], vert_textures=verts_rgb.to(device))

    #### Set viewpoints_name in load_val_dataset ####
    images, silhouettes, rotations, translations, K, transf = load_val_dataset(path_str ='data',device=device, n_images=config['training']['n_image_per_batch'], viewpoints_name = 'behind')
    #images, silhouettes, rotations, translations, K, transf = load_dataset(path_str ='data',device=device, n_images=config['training']['n_image_per_batch'])
    
    camera_settings = pytorch_camera(config['rendering']['rgb']['image_size'], K)
    images = images.cpu()


    pos_encode_weight = torch.cat(tuple(torch.eye(3) * (1.5**i) for i in range(0,14,1)), dim=0) #######
    pos_encode_out_weight = torch.cat(tuple( torch.tensor([1.0/(1.3**i)]*3) for i in range(0,14,1)), dim=0) #######
    
    shape_net = ShapeNet(velocity_mlp= Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),
                        MLP(pos_encode_weight.shape[0]*2, [256,256,256,3], ['lrelu','lrelu','lrelu','tanh']),  
                        ), T=config['sampling']['T']).to(device)
    N_IT =  config['training']['n_iterations']-1
    
    #loaded_data = load_models(f'./out/{file}/checkpoint_{N_IT}')
    #print("Loss",loaded_data['loss'])
    #loaded_shape_net_state_dict = loaded_data['shape_net']
    #shape_net.load_state_dict(loaded_shape_net_state_dict)
    #shape_net.eval()
    
    checkpoint = torch.load(f'./out/{file}/checkpoint_{N_IT}')
    shape_net.load_state_dict(checkpoint['shape_net'])
    shape_net.eval()

    light_dirs=None
    n_images = len(images)

    ## sample mesh from neural nets
    init_mesh = rand_ico_sphere(config['sampling']['ico_sphere_level'], device=device)
    s = init_mesh.verts_packed()

    if (config['loss']['lambda_velocity']==0) or (config['loss']['alpha'] == 0) or config['training']['compute_velocity_seperately']:
        x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
    else:
        x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 2)

    x = x_arr[-1]

    image_size=config['rendering']['rgb']['image_size']
    batch_idx = torch.randperm(n_images)[:config['training']['n_image_per_batch']]
    loss_image, loss_silhouette, loss_velocity = 0, 0, 0

    is_render_checkpoint = True
    if(is_render_checkpoint):
        col_count = config['training']['render_cols']
        img_grid_width = int(col_count * image_size)
        img_grid_height = int(n_images / col_count * image_size)

        gt_grid = np.zeros((img_grid_height, img_grid_width, 3), dtype=np.uint16)
        prd_grid = np.zeros((img_grid_height, img_grid_width, 3), dtype=np.uint16)

        if(config['loss']['lambda_silhouette'] != 0):
            gt_sil_grid = np.zeros((img_grid_height, img_grid_width, 1), dtype=np.uint16)
            prd_sil_grid = np.zeros((img_grid_height, img_grid_width, 1), dtype=np.uint16)

    for i in batch_idx:
        gt_image, gt_silhouette = images[i:i+1], silhouettes[i:i+1]
        
        light_pose = None
        if light_dirs is not None:
            light_pose = light_dirs[i:i+1]

        translation = translations[i:i+1]

        prd_image = render_mesh(mesh, 
                modes='image_ct', #######
                L0=config['rendering']['rgb']['L0'],
                rotations=rotations[i:i+1], 
                translations=translation, 
                image_size=image_size, 
                blur_radius=config['rendering']['rgb']['blur_radius'], 
                faces_per_pixel=config['rendering']['rgb']['faces_per_pixel'], 
                device=device, background_colors=None, light_poses=light_pose, materials=None, camera_settings=camera_settings,
                sigma=config['rendering']['rgb']['sigma'], gamma=config['rendering']['rgb']['gamma'])[...,:3]

        if(is_render_checkpoint):
            grid_x_start = i.item() // col_count * image_size
            grid_x_end = grid_x_start + image_size
            grid_y_start = (i.item() % col_count)* image_size
            grid_y_end = grid_y_start + image_size

            img = (prd_image[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
            prd_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = img

            img = (gt_image[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
            gt_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = img
        
        if config['loss']['lambda_image'] != 0:
            max_intensity = config['rendering']['rgb']['max_intensity'] #* (np.random.rand()+1)
            loss_tmp = clipped_mae(gt_image.cuda(), prd_image) / config['training']['n_image_per_batch']

            #print(float(loss_tmp))
            # loss_tmp = clipped_mae(gt_image.cuda().clamp_max(max_intensity), prd_image, max_intensity) / config['training']['n_image_per_batch']
            # print("loss",loss_tmp)
            #(loss_tmp * config['loss']['lambda_image']).backward(retain_graph=True)
            loss_image += loss_tmp 
            

        if config['loss']['lambda_silhouette'] != 0:
            prd_silhouette = render_mesh(mesh, 
                    modes='silhouette', 
                    L0=None,
                    rotations=rotations[i:i+1], 
                    translations=translations[i:i+1], 
                    image_size=config['rendering']['silhouette']['image_size'], 
                    blur_radius=config['rendering']['silhouette']['blur_radius'], 
                    faces_per_pixel=config['rendering']['silhouette']['faces_per_pixel'], 
                    device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings,
                    sigma=config['rendering']['silhouette']['sigma'], gamma=config['rendering']['silhouette']['gamma'])

            prd_silhouette = torch.unsqueeze(prd_silhouette,0)
            gt_silhouette = torch.unsqueeze(gt_silhouette,3)

            if(is_render_checkpoint):
                img = (prd_silhouette[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
                prd_sil_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = img

                img = (gt_silhouette[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
                gt_sil_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = img

            loss_tmp = mse(gt_silhouette.cuda(), prd_silhouette) / config['training']['n_image_per_batch']
            #(loss_tmp * config['loss']['lambda_silhouette']).backward(retain_graph=True)
            loss_silhouette += loss_tmp
            

    path = join('./out',file,'validation')
    if os.path.exists(path)== False:
        os.mkdir(path)
    cv2.imwrite(f"{path}/gt.png", gt_grid)
    if(config['loss']['lambda_silhouette'] != 0):
        cv2.imwrite(f"{path}/gt_sil.png", gt_sil_grid)

    if (is_render_checkpoint):
        cv2.imwrite(f"{path}/prd_val" + ".png", prd_grid)
        if(config['loss']['lambda_silhouette'] != 0):
            cv2.imwrite(f"{path}/prd_sil_val" + ".png", prd_sil_grid)
    
    if config['loss']['lambda_velocity'] == 0:
        pass
    elif (config['loss']['alpha'] == 0) or (not config['training']['compute_velocity_seperately']):
        loss_tmp = velocity_loss(v_arr, d2v_d2s_arr, config['loss']['alpha'])
        #(loss_tmp * config['loss']['lambda_velocity']).backward(retain_graph=True)
        loss_velocity = loss_tmp
    # else:
    #     ico_sphere = rand_ico_sphere(config['training']['sampling_lvl_for_vel_loss'], device=device)
    #     if init_mesh is not None:
    #         n_verts = ico_sphere.verts_packed().shape[0]
    #         s = pytorch3d.ops.sample_points_from_meshes(init_mesh, n_verts).reshape(n_verts, 3)
    #     else:
    #         s = ico_sphere.verts_packed()
        
    #     n_pts_total = s.shape[0]
    #     for _s in torch.split(s, config['training']['n_pts_per_split'], dim=0):
    #         n_pts = _s.shape[0]
    #         _x_arr, _v_arr, _dv_ds_arr, _d2v_d2s_arr = shape_net(_s, 2)
    #         loss_tmp = velocity_loss(_v_arr, _d2v_d2s_arr, config['loss']['alpha']) * n_pts / n_pts_total
    #         #(loss_tmp * config['loss']['lambda_velocity']).backward(retain_graph=False)
    #         loss_velocity += loss_tmp.detach()

    if config['loss']['lambda_edge'] != 0:
        loss_edge = pytorch3d.loss.mesh_edge_loss(mesh)
        #(loss_edge * config['loss']['lambda_edge']).backward(retain_graph=True)
    else:
        loss_edge = 0

    if config['loss']['lambda_normal_consistency'] != 0:
        loss_normal_consistency = pytorch3d.loss.mesh_normal_consistency(mesh)
        #(loss_normal_consistency * config['loss']['lambda_normal_consistency']).backward(retain_graph=True)
    else:
        loss_normal_consistency = 0

    if config['loss']['lambda_laplacian_smoothing'] != 0:
        loss_laplacian_smoothing = pytorch3d.loss.mesh_laplacian_smoothing(mesh)
        #(loss_laplacian_smoothing * config['loss']['lambda_laplacian_smoothing']).backward(retain_graph=True)
    else:
        loss_laplacian_smoothing = 0


    loss =  loss_image * config['loss']['lambda_image'] + \
            loss_silhouette * config['loss']['lambda_silhouette'] + \
            loss_velocity * config['loss']['lambda_velocity']  + \
            loss_edge * config['loss']['lambda_edge'] + \
            loss_normal_consistency * config['loss']['lambda_normal_consistency']+ \
            loss_laplacian_smoothing * config['loss']['lambda_laplacian_smoothing']
    
    path = join('./out',file,'validation','loss.pckl')
    f = open(path, 'wb')
    pickle.dump(loss, f)
    pickle.dump(loss_image, f)
    pickle.dump(loss_velocity, f)
    f.close()

    print("Loss", float(loss))
    print("loss_image", float(loss_image))
    print("loss_velocity", float(loss_velocity))
    #return (float(loss), float(loss_image), float(loss_silhouette), float(loss_velocity), float(loss_edge), float(loss_normal_consistency), float(loss_laplacian_smoothing))
    



    

    
    