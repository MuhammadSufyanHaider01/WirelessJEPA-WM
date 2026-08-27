#!/bin/bash
#### sbatch job-script.slurm ####
####### Reserve computing resources #############
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=4
# SBATCH --time=3:00:00
# SBATCH --mem=32G
# SBATCH --partition=gpu-h100
# SBATCH --gres=gpu:1

# SBATCH -p gpu-h100
#SBATCH -p gpu-v100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1      # one H100 per rank
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:20:00

    
####### Set environment variables ###############
source ~/software/init-conda
conda activate jepa38
cd /home/omar.mashaal1/JEPA3/CNN-JEPA/

####### Run your script #########################

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rf --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/column.ckpt --sample 50
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rml --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/column.ckpt --sample 50

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/last.ckpt --sample 50

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rf --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/column.ckpt --sample 100
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rml --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/column.ckpt --sample 100

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/last.ckpt --sample 500



# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task jamming --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/row.ckpt --sample 500

# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task inter --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/epoch=398-step=606081.ckpt --sample 500







# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rf --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task rml --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task deepbeam --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task jamming --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
PYTHONPATH=. python pretrain/eval_ijepacnn.py --task tpi --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/column.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task radar --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task craw --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500
# PYTHONPATH=. python pretrain/eval_ijepacnn.py --task inter --checkpoint /home/omar.mashaal1/JEPA3/CNN-JEPA/artifacts/random.ckpt --sample 500











