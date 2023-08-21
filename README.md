


### Instructions for running the code
 
 1. Create and activate pytorch3d v0.3 virtual environment `conda create -f p3d3.yml; conda activate p3d3`
 2. Set up the config.yaml in the folder 'config' with the preferred hyper-parameters used for training and validation processes.
 3. Add more '.yaml' files in the folder 'config' to run multiple training sessions 
 4. Create training and validating datasets with `python create_dataset.py`
 5. Run code with `python train.py`
 6. Results will be saved in folder 'out'
 7. Redirect to folder 'out' and represent the results with `python; import tensorboard; tensorboard --logdir=tensorboard`
 

