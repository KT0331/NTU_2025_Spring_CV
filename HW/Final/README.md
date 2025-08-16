# Iris Recognition Challenge
Assigned by GANZIN Technology.  
## Environment
```sh
sh init.sh <env_name>
```
env_name: Environment Name  

## Generate Patch Dataset
```sh
sh gen_train_pair_data.sh
sh patch_gen.sh
```

## Training
```sh
sh train.sh <GPU ID> <Mode>
```
Mode: 'patch', 'seg', or leave it empty  

## Inference
```sh
sh inference.sh <GPU ID> <Mode>
```
Mode: 'patch', or 'seg', or leave it empty  
