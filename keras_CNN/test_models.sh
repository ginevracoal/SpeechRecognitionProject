#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:kepler:2
#SBATCH --time=4:00:00
#SBATCH --partition=gll_usr_gpuprod
#SBATCH --account=uts18_bortoldl_0

. source virtualenv_2/bin/activate
python /galileo/home/userexternal/gcarbone/group/keras_CNN/test_models.py model_1 > group/keras_CNN/trained_models/model_1.out
python /galileo/home/userexternal/gcarbone/group/keras_CNN/test_models.py model_2 > group/keras_CNN/trained_models/model_2.out