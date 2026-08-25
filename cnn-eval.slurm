#!/bin/bash
#### sbatch job-script.slurm ####
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
    
####### Set environment variables ###############
source ~/software/init-conda
conda activate cnn-jepa
export WANDB_API_KEY="bada5503078ed952509582876bc0cd850cae8532"
cd ~/ucalgary/iqfm-jepa/CNN-JEPA

####### Run your script #########################
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rf --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_0/epoch=100-step=153419.ckpt

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rml --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_0/epoch=100-step=153419.ckpt

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_0/epoch=100-step=153419.ckpt

# Best model now: 100 epoch 
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskmulti-block_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_2/last.ckpt

PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --seed 42 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --seed 43 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --seed 44 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --seed 45 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task aoa --seed 46 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt

PYTHONPATH=. python pretrain/eval_ijepacnn.py --task mod --seed 42 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task mod --seed 43 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task mod --seed 44 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task mod --seed 45 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task mod --seed 46 --checkpoint ./artifacts/pretrain_lightly/ijepacnn_iqfm/I-JEPA_iqfm_shufflenet_v2_x0_5_torchvision_IN1kCFG_predL3K3_Maskrandom_lr0.005_wd0.005_wu0.0_bs256_4GPU/version_3/last.ckpt