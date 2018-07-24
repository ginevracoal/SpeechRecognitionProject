#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:kepler:2
#SBATCH --time=4:00:00
#SBATCH --partition=gll_usr_gpuprod
#SBATCH --account=uts18_bortoldl_0

. virtualenv_2/bin/activate

## 20 epochs, job 102242
# python /galileo/home/userexternal/gcarbone/group/keras_CNN/test_models.py model_2

## 20 epochs, job 102261, killato perché troppo lento...
# python /galileo/home/userexternal/gcarbone/group/keras_CNN/test_models.py model_3

## 10 epochs, job 102274 
python /galileo/home/userexternal/gcarbone/group/keras_CNN/test_models.py model_4
