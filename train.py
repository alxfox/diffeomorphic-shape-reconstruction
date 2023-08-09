import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from pytorch3d.io import load_ply
import pytorch3d
import torch
import numpy as np
from Loss import velocity_loss, clipped_mae
from torch.nn.functional import mse_loss as mse
from Render import render_mesh
from Model import MLP, PositionEncoding, ResNet, Sequential, ShapeNet, BRDFNet
from os import listdir
from os.path import isfile, join
from utils import manual_seed, rand_ico_sphere, save_models, load_models
from Meshes import Meshes
from tqdm import tqdm
import trimesh
from Model import MLP, PositionEncoding, Sequential, ShapeNet, BRDFNet
from itertools import chain
from utils import dotty, pytorch_camera, random_crop
import os
from diligent import diligent_eval_chamfer
import pickle
import yaml
import uuid
from torch.utils.tensorboard import SummaryWriter
from validation import validation

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

@torch.no_grad()
def sample_mesh(config, shape_net, brdf_net, init_mesh=None, normal_net=None):
    config = dotty(config)
    if init_mesh is None:
        init_mesh = rand_ico_sphere(config['sampling']['ico_sphere_level'], device=device)
            
    s = init_mesh.verts_packed()

    x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
    theta_x = brdf_net(s)

    x = x_arr[-1]

    faces = init_mesh.faces_packed()
    mesh = Meshes(verts=[x], faces=[faces], vert_textures=[theta_x])

    return mesh

def clamp_vertex_grad(grad, thre):
    ret = grad + 0
    ret[torch.logical_not(torch.abs(ret) < thre)] = 0
    return ret

def train(config, device, images, silhouettes, cubes, rotations, translations, shape_net, brdf_net, optimizer, n_iterations, light_dirs=None, call_back=None, init_mesh=None,  camera_settings=None,  camera_settings_silhouette=None):
    n_images = len(images)

    null_init = init_mesh is None

    writer = SummaryWriter(config['experiment_path']+"/tensorboard")
    writer.add_hparams({ "lr": config["training"]["lr"], "T":config["sampling"]["T"],
                        "ico_sphere_level": config["sampling"]["ico_sphere_level"] },{"value":0},
                        run_name=os.path.dirname(os.path.realpath(__file__)) + os.sep + config['experiment_path']+"/tensorboard")
    
    early_stopper = EarlyStopper(patience=config['validation']['early_stopping_patience'], min_delta=0.0)

    def closure():
        is_render_checkpoint = N_IT % config['training']['render_interval'] == 0 or N_IT == n_iterations - 1
        nonlocal init_mesh

        if null_init:
            init_mesh = rand_ico_sphere(config['sampling']['ico_sphere_level'], device=device)
        s = init_mesh.verts_packed()

        if (config['loss']['lambda_velocity']==0) or (config['loss']['alpha'] == 0) or config['training']['compute_velocity_seperately']:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
        else:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 2)

        vertices = x_arr[-1]

        if config.get('training.vertex_grad_clip', None) is not None:
            hook = vertices.register_hook(lambda grad: clamp_vertex_grad(grad, config['training']['vertex_grad_clip']))

        theta_x = brdf_net(s)

        faces = init_mesh.faces_packed()

        mesh = Meshes(verts=[vertices], faces=[faces], vert_textures=[theta_x])

        batch_idx = torch.randperm(n_images)[:config['training']['n_image_per_batch']]
        loss_image, loss_silhouette, loss_velocity = 0, 0, 0

        crop = config['rendering']['rgb']['crop']
        crop_ratio = config['rendering']['rgb']['crop_ratio']
        image_size=config['rendering']['rgb']['image_size']

        if crop:
            crop_image_size = int(image_size * crop_ratio)
            max_val = image_size - crop_image_size
            x = np.random.randint(0, max_val)
            y = np.random.randint(0, max_val)
        
        if is_render_checkpoint:
            col_count = config['training']['render_cols']
            img_grid_width = int(col_count * image_size)
            img_grid_height = int(n_images / col_count * image_size)

            gt_grid = np.zeros((img_grid_height, img_grid_width, 3), dtype=np.uint16)
            prd_grid = np.zeros((img_grid_height, img_grid_width, 3), dtype=np.float32)

            # gt_sil_grid = np.zeros((img_grid_height, img_grid_width, 1), dtype=np.uint16)
            prd_sil_grid = np.zeros((img_grid_height, img_grid_width, 1), dtype=np.float32)

        for i in batch_idx:
            gt_image, gt_silhouette = images[i:i+1], silhouettes[i:i+1]
            gt_cubes = cubes[i:i+1]

            # NeRF_bgc is the image of cubes
            NeRF_bgc = gt_cubes

            light_pose = None
            if light_dirs is not None:
                light_pose = light_dirs[i:i+1]

            translation = translations[i:i+1]

            # Check if the rendering should be on a subpart of the image
            if crop:
                cropped_gt_image = random_crop(gt_image[0], crop_image_size, x, y)

            prd_image = render_mesh(mesh, 
                    modes='image_ct', #######
                    L0=config['rendering']['rgb']['L0'],
                    rotations=rotations[i:i+1], 
                    translations=translation, 
                    image_size=image_size, 
                    blur_radius=config['rendering']['rgb']['blur_radius'], 
                    faces_per_pixel=config['rendering']['rgb']['faces_per_pixel'], 
                    device=device, background_colors=None, NeRF_bgc=NeRF_bgc, light_poses=light_pose, materials=None, camera_settings=camera_settings,
                    sigma=config['rendering']['rgb']['sigma'], gamma=config['rendering']['rgb']['gamma'])[...,:3]

            if crop:
                cropped_prd_image = random_crop(prd_image[0], crop_image_size, x, y)

            if(is_render_checkpoint):
                grid_x_start = i.item() // col_count * image_size
                grid_x_end = grid_x_start + image_size
                grid_y_start = (i.item() % col_count)* image_size
                grid_y_end = grid_y_start + image_size

                prd_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = prd_image[0].detach().cpu().numpy()
            
            if(N_IT == 0):
                img = (gt_image[0]*(256**2-1)).detach().cpu().numpy().astype(np.uint16)
                gt_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = img

            if config['loss']['lambda_image'] != 0:
                if crop: 
                    loss_tmp = clipped_mae(cropped_gt_image.cuda(), cropped_prd_image) / config['training']['n_image_per_batch']
                    (loss_tmp * config['loss']['lambda_image']).backward(retain_graph=True)
                    loss_image += loss_tmp.detach()
                else: 
                    loss_tmp = clipped_mae(gt_image.cuda(), prd_image) / config['training']['n_image_per_batch']
                    (loss_tmp * config['loss']['lambda_image']).backward(retain_graph=True)
                    loss_image += loss_tmp.detach()

            if config['loss']['lambda_silhouette'] != 0:
                prd_silhouette = render_mesh(mesh, 
                        modes='silhouette', 
                        L0=None,
                        rotations=rotations[i:i+1], 
                        translations=translations[i:i+1], 
                        image_size=config['rendering']['silhouette']['image_size'], 
                        blur_radius=config['rendering']['silhouette']['blur_radius'], 
                        faces_per_pixel=config['rendering']['silhouette']['faces_per_pixel'], 
                        device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings_silhouette,
                        sigma=config['rendering']['silhouette']['sigma'], gamma=config['rendering']['silhouette']['gamma'])

                prd_silhouette = torch.unsqueeze(prd_silhouette,0)
                gt_silhouette = torch.unsqueeze(gt_silhouette,3)

                if(is_render_checkpoint):
                    prd_sil_grid[grid_x_start:grid_x_end, grid_y_start:grid_y_end] = prd_silhouette[0].detach().cpu().numpy()

                loss_tmp = mse(gt_silhouette.cuda(), prd_silhouette) / config['training']['n_image_per_batch']
                (loss_tmp * config['loss']['lambda_silhouette']).backward(retain_graph=True)
                loss_silhouette += loss_tmp.detach()
        
        if config['loss']['lambda_velocity'] == 0:
            pass
        elif (config['loss']['alpha'] == 0) or (not config['training']['compute_velocity_seperately']):
            loss_tmp = velocity_loss(v_arr, d2v_d2s_arr, config['loss']['alpha'])
            (loss_tmp * config['loss']['lambda_velocity']).backward(retain_graph=True)
            loss_velocity = loss_tmp.detach()
        else:
            ico_sphere = rand_ico_sphere(config['training']['sampling_lvl_for_vel_loss'], device=device)
            if init_mesh is not None:
                n_verts = ico_sphere.verts_packed().shape[0]
                s = pytorch3d.ops.sample_points_from_meshes(init_mesh, n_verts).reshape(n_verts, 3)
            else:
                s = ico_sphere.verts_packed()
            
            n_pts_total = s.shape[0]
            for _s in torch.split(s, config['training']['n_pts_per_split'], dim=0):
                n_pts = _s.shape[0]
                _x_arr, _v_arr, _dv_ds_arr, _d2v_d2s_arr = shape_net(_s, 2)
                loss_tmp = velocity_loss(_v_arr, _d2v_d2s_arr, config['loss']['alpha']) * n_pts / n_pts_total
                (loss_tmp * config['loss']['lambda_velocity']).backward(retain_graph=False)
                loss_velocity += loss_tmp.detach()

        if config['loss']['lambda_edge'] != 0:
            loss_edge = pytorch3d.loss.mesh_edge_loss(mesh)
            (loss_edge * config['loss']['lambda_edge']).backward(retain_graph=True)
        else:
            loss_edge = 0

        if config['loss']['lambda_normal_consistency'] != 0:
            loss_normal_consistency = pytorch3d.loss.mesh_normal_consistency(mesh)
            (loss_normal_consistency * config['loss']['lambda_normal_consistency']).backward(retain_graph=True)
        else:
            loss_normal_consistency = 0

        if config['loss']['lambda_laplacian_smoothing'] != 0:
            loss_laplacian_smoothing = pytorch3d.loss.mesh_laplacian_smoothing(mesh)
            (loss_laplacian_smoothing * config['loss']['lambda_laplacian_smoothing']).backward(retain_graph=True)
        else:
            loss_laplacian_smoothing = 0

        # print("Image loss: ", loss_image)

        loss =  loss_image * config['loss']['lambda_image'] + \
                loss_silhouette * config['loss']['lambda_silhouette'] + \
                loss_velocity * config['loss']['lambda_velocity']  + \
                loss_edge * config['loss']['lambda_edge'] + \
                loss_normal_consistency * config['loss']['lambda_normal_consistency']+ \
                loss_laplacian_smoothing * config['loss']['lambda_laplacian_smoothing']

        return mesh.detach(), (float(loss), float(loss_image), float(loss_silhouette), float(loss_velocity), float(loss_edge), float(loss_normal_consistency), float(loss_laplacian_smoothing)), (prd_grid, ), (gt_grid, )
    
    pbar = tqdm(range(n_iterations))

    for N_IT in pbar:
        optimizer.zero_grad()
        mesh, losses, rendered_images, gt_images = closure()
        if(N_IT % config['training']['checkpoint_interval'] == 0 or N_IT == n_iterations-1):
            with torch.no_grad():
                save_models(f'{config["experiment_path"]}/{checkpoint_name}_{N_IT}', brdf_net=brdf_net, shape_net=shape_net, 
                            optimizer=optimizer, meta=dict(loss=losses[0], params=config))
                writer.add_mesh("Mesh/Pred", mesh.verts_packed().unsqueeze(0), faces=mesh.faces_packed().unsqueeze(0), global_step=N_IT)
        
        writer.add_image('Image/Pred', (rendered_images[0]*255).clip(0,255).astype(np.uint8), dataformats="HWC", global_step=N_IT)
            
        if(N_IT == 0):
            writer.add_image('Image/GT', (gt_images[0]/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)
        
        writer.add_scalar('Loss/train', losses[0], N_IT)
        optimizer.step()
        pbar.set_description('|'.join(f'{l:.2e}' for l in losses).replace('e', '').replace('|', ' || ', 1))

        if call_back is not None:
            call_back(mesh, losses[0])

        if(N_IT % config['validation']['interval'] == 0 or N_IT == n_iterations-1):
            print("Starting validation...")
            
            shape_net.eval()
            brdf_net.eval()
            with torch.no_grad():
                mesh = sample_mesh(config, shape_net, brdf_net)#init_mesh=init_mesh, **params)

                angle = config['validation']['angle']

                if angle == 'behind':
                    loss_val, gt, prd =validation(config, N_IT, mesh, shape_net,angle = 'behind')
                    writer.add_scalar('Loss/val_behind', float(loss_val), N_IT)
                    writer.add_image('Image/Pred_behind', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)
                elif angle == 'above':
                    loss_val , gt, prd=validation(config, N_IT, mesh, shape_net,angle = 'above')
                    writer.add_scalar('Loss/val_above', float(loss_val), N_IT)
                    writer.add_image('Image/Pred_above', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)
                elif angle == 'below':
                    loss_val , gt, prd= validation(config, N_IT, mesh, shape_net,angle = 'below')
                    writer.add_scalar('Loss/val_below', float(loss_val), N_IT)
                    writer.add_image('Image/Pred_below', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)
                else :
                    loss_val1 , gt, prd= validation(config, N_IT, mesh, shape_net,angle = 'behind')
                    writer.add_scalar('Loss/val_behind', float(loss_val1), N_IT)
                    writer.add_image('Image/Pred_behind', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)

                    loss_val2 , gt, prd= validation(config, N_IT, mesh, shape_net,angle = 'above')
                    writer.add_scalar('Loss/val_above', float(loss_val2), N_IT)
                    writer.add_image('Image/Pred_above', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)

                    loss_val3 , gt, prd= validation(config, N_IT, mesh, shape_net,angle = 'below')
                    writer.add_scalar('Loss/val_below', float(loss_val3), N_IT)
                    writer.add_image('Image/Pred_below', (prd/256).astype(np.uint8), dataformats="HWC", global_step=N_IT)

                    loss_val = float(loss_val1)+float(loss_val2)+float(loss_val3)
                    loss_val = float(loss_val1)+float(loss_val2)+float(loss_val3)
                
                    loss_val = float(loss_val1)+float(loss_val2)+float(loss_val3)    
                

                if config['validation']['early_stopping'] == True and early_stopper.early_stop(loss_val):
                    print("Early stopping at iter:", N_IT)
                    break
            
            print("Validation completed!")

            shape_net.train()
            brdf_net.train()

    #call_back(end=True)

    return losses, N_IT

def call_back(mesh=None, loss=None, end=False):

    if end:
        call_back.history = []
        call_back.video_writer.release()
        call_back.video_writer = None
        return

    if not hasattr(call_back, 'history'):
        call_back.history = []
    call_back.history.append(float(loss))

    # if min(call_back.history) == loss:
    #     save_models(f'./out/{checkpoint_name}', brdf_net=brdf_net, shape_net=shape_net, 
    #                 optimizer=optimizer, meta=dict(loss=loss, params=dict(params)))
    
    # with torch.no_grad():
    #     frame = render_mesh(mesh, 
    #                         modes='image_ct', #######
    #                         rotations=R[0:1], 
    #                         translations=T[0:1], 
    #                         L0=params['rendering']['rgb']['L0'],
    #                         image_size=params['rendering']['rgb']['image_size'], 
    #                         blur_radius=params['rendering']['rgb']['blur_radius'], 
    #                         faces_per_pixel=params['rendering']['rgb']['faces_per_pixel'], 
    #                         device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings)[...,:3]
        
    #     frame = torch.cat(list(f for f in frame), dim=1)
    #     frame = np.clip((frame * 255 / params['rendering']['rgb']['max_intensity']).cpu().numpy(), 0, 255).astype(np.uint8) #######
    # if not hasattr(call_back, 'video_writer') or call_back.video_writer is None:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     call_back.video_writer = cv2.VideoWriter(f'./out/{checkpoint_name}.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))
    # img = (frame*(256**2-1)).astype(np.uint16)
    # cv2.imwrite(f"./out/render.png", img)
    # call_back.video_writer.write(frame)

if __name__ == '__main__':
    config_list = [join('./config',f) for f in listdir('./config') if isfile(join('./config', f))]

    print("Configs to run: ", config_list)

    for conf in config_list:
        config = yaml.safe_load(open(conf))
        if(not config.get('experiment_path')):
            name = config.get('name')
            config['experiment_name'] = (name + "_" if name else "") + str(uuid.uuid4())
            config['experiment_path'] = 'out/' + config['experiment_name']
        print("Saving the results to " + config['experiment_path'] + "\n")

        try:
            os.makedirs(config['experiment_path'])
        except OSError as error:
            print(error)

        if(config['rendering']['rgb']['L0']=='None'):
            f = open('store.pckl', 'rb')
            config['rendering']['rgb']['L0'] = pickle.load(f).item()
            f.close()

        print("using config:\n", config, "\n")
        with open(f'{config["experiment_path"]}/config.yaml', 'w') as file:
            yaml.dump(config, file, indent=4, sort_keys=False)

        device = torch.device('cuda:0')
        manual_seed(config['training']['rand_seed'])
        checkpoint_name = 'checkpoint' #######  'diligent_reading'
        
        from dataloader import load_dataset
        images, silhouettes, cubes, R, T, K, transf = load_dataset('data', n_images=config['training']['n_image_per_batch'], device=device)
        
        images = images.cpu()

        pos_encode_weight = torch.cat(tuple(torch.eye(3) * (1.5**i) for i in range(0,14,1)), dim=0) #######
        pos_encode_out_weight = torch.cat(tuple( torch.tensor([1.0/(1.3**i)]*3) for i in range(0,14,1)), dim=0) #######
        
        shape_net = ShapeNet(velocity_mlp= Sequential(
                            PositionEncoding(pos_encode_weight, pos_encode_out_weight),
                            MLP(pos_encode_weight.shape[0]*2, [256,256,256,3], ['lrelu','lrelu','lrelu','tanh']),  
                            ), T=config['sampling']['T']).to(device)

        brdf_net = BRDFNet(Sequential(
                            PositionEncoding(pos_encode_weight, pos_encode_out_weight),  
                            MLP(pos_encode_weight.shape[0]*2, [256]*5+[config['n_lobes']*3+3], ['lrelu']*5+['none']),    
                            ), constant_fresnel=True).to(device)

        optimizer = torch.optim.Adam(list(shape_net.parameters())+list(brdf_net.parameters()), lr=config['training']['lr'])

        camera_settings_silhouette = pytorch_camera(config['rendering']['silhouette']['image_size'], K)
        camera_settings = pytorch_camera(config['rendering']['rgb']['image_size'], K)

        verts, faces = load_ply("data/mesh.ply")
        init_mesh = Meshes(verts=[verts], faces=[faces]).to(device)

        print("Starting training...")

        losses, N_IT = train(config, device, images, silhouettes, cubes, R, T, shape_net, brdf_net, optimizer, config['training']['n_iterations'],  
                call_back=call_back, 
                light_dirs=None,
                camera_settings = camera_settings,
                camera_settings_silhouette=camera_settings_silhouette,
                init_mesh=None,
                )
        
        print("Training completed!")
        
        # Load the checkpoint model
        load_models(f'{config["experiment_path"]}/{checkpoint_name}_{N_IT}', brdf_net=brdf_net, shape_net=shape_net, optimizer=optimizer)
        
        mesh = sample_mesh(config, shape_net, brdf_net)#init_mesh=init_mesh, **params)
        trimesh.Trimesh((mesh.verts_packed().detach()).cpu().numpy(), 
                        mesh.faces_packed().cpu().numpy()).export(f'{config["experiment_path"]}/{checkpoint_name}.obj')

        # compile_video(mesh, f'{checkpoint_name}.mp4', distance=2, render_mode='image_ct', **params)