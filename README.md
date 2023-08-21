### Instructions for running the code
 
 1. Create and activate pytorch3d v0.3 virtual environment `conda create -f p3d3.yml; conda activate p3d3`
 2. Set up the config.yaml in the folder 'config' with the preferred hyper-parameters used for training and validation processes.
 3. Add more '.yaml' files in the folder 'config' to run multiple training sessions 
 4. Create training and validating datasets with `python create_dataset.py`
 5. Run code with `python train.py`
 6. Redirect to folder 'out' and represent the results with `python; import tensorboard; tensorboard --logdir=tensorboard` and open tensorboard with `http://localhost:6006/` in a browser.

### Description of files for dataset creation
#### Camera settings
- cameras1.npz - Viewpoints from the front of the mesh
- cameras_behind.npz - Viewpoints from all sides of the mesh
- cameras_above.npz - Viewpoints from above the mesh
- cameras_below.npz - Viewpoints from below the mesh

#### Meshes 
- mesh.ply - bunny
- cubemesh.ply - mesh with bunny and cubes for half-front rotation
- cubes.ply -  mesh with cubes used when background is provided by our renderer when performing half-front rotation  
- rotation_scene.ply - mesh with bunny and cubes for full rotation
- rotation_cubes.ply - mesh with cubes used when background is provided by our renderer when performing full rotation 



 

