#!/bin/bash
#### sbatch job-script.slurm ####
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
### SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
    
####### Set environment variables ###############
source ~/software/init-conda
conda activate cnn-jepa
export WANDB_API_KEY="bada5503078ed952509582876bc0cd850cae8532"
cd ~/ucalgary/iqfm-jepa/CNN-JEPA

####### Run your script #########################
# PYTHONPATH=. python pretrain/train_ijepacnn.py --config-name ijepacnn_imagenet100.yaml 

# PYTHONPATH=. python pretrain/train_ijepacnn.py --config-name ijepacnn_iqfm_shufflenet.yaml
PYTHONPATH=. python pretrain/train_ijepacnn.py --config-name ijepacnn_iqfm_random.yaml