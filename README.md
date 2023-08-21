### Instructions for running the code
1. Create and activate the environment provided using `conda create -f p3d3.yml; conda activate p3d3`
2. Create training and validation datasets with `python create_dataset.py`. If desired, the color of the cubes must be changed at this step.
3. Set up the config.yaml in the folder 'config' with the preferred hyper-parameters used for training and validation processes.
4. Add more '.yaml' files in the folder 'config' to run multiple training sessions 
 5. Run training with `python train.py`

Results are placed in the `out/` folder

The results may be viewed in a web browser at `http://localhost:6006/` by using `tensorboard --logdir=out`.

### Description of provided files for dataset creation
#### Camera settings
- cameras1.npz - Viewpoints from the front of the mesh
- cameras_behind.npz - Viewpoints from all sides of the mesh
- cameras_above.npz - Viewpoints from above the mesh
- cameras_below.npz - Viewpoints from below the mesh

#### Meshes 
- mesh.ply - mesh with bunny
- cubemesh.ply - mesh with both bunny and cubes for half-front rotation
- cubes.ply -  only cubes for half-front rotation  
- rotation_scene.ply - mesh with bunny and cubes for full rotation
- rotation_cubes.ply - mesh with cubes used when background is provided by our renderer when performing full rotation 
