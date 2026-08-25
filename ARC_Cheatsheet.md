ARC running guide: https://rcs.ucalgary.ca/ARC_Cluster_Guide


arc.nodes
# to see ranking
squeue 
squeue --partition=gpu-h100 --state=PD --sort=-P
squeue --sort=-P | grep -B 10 "$USER"


squeue -u $USER
sshare -u $USER

sprio -j 36166152
sprio --sort=-priority
arc.job-info 35518134
scancel 35518134
sacct -j <jobid> --format=JobID,JobName,Elapsed,Start,End #check elapse time

sbatch job-script.slurm

REQUEST 1 GPU from 1 node belonging to the gpu-a100 partition with 4 GB of RAM for 1 hour. Generic resource scheduling (--gres) is used to request for GPU resources.

salloc --partition=legacy --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=05:00:00
salloc --partition=bigmem --nodes=1 --ntasks=1 --cpus-per-task=40 --mem=32G --time=05:00:00
salloc --partition=cpu2019-bf05 --nodes=1 --ntasks=1 --cpus-per-task=40 --mem=64G --time=03:00:00
salloc --mem=128G -t 05:00:00 -p gpu-v100 --gres=gpu:1 --cpus-per-task=40
salloc --mem=32G -t 05:00:00 -p gpu-v100 --gres=gpu:1 --cpus-per-task=4

[tannistha.nandi@arc ~]$ salloc --mem=4G -t 02:00:00 -p gpu-v100 --gres=gpu:1
salloc: Granted job allocation 6758015
salloc: Waiting for resource configuration
salloc: Nodes fc4 are ready for job
[tannistha.nandi@fc4 ~]$ 

EXIT: 
[tannistha.nandi@fg3 ~]$ exit
[tannistha.nandi@fg3 ~]$ salloc: Relinquishing job allocation 6760460


source ~/software/init-conda
conda activate cnn-jepa
cd iqfm-jepa/CNN-JEPA


Set up MongoDB: 
apptainer pull mongo_latest.sif docker://mongo:6
mkdir -p $HOME/mongodb_data

Step 3: Run MongoDB inside the container
nohup apptainer exec mongo_latest.sif mongod --dbpath $HOME/mongodb_data --port 27018 > mongo.log 2>&1 &

You can check that it’s running with:
ps aux | grep mongod

To stop it:
pkill mongod


RUN ZCORE:

python zeroshot_coreset_selection.py --dataset eurosat10 --data_dir ./data_core --results_dir ./results_core --embedding shuffle resnet18 --num_workers 40

python zeroshot_coreset_selection.py --dataset iqfm --h5_file_path ./Dataset/train_256_100_256_22.h5 --data_dir ./Dataset --checkpoint_path ./model/epoch=398-step=606081.ckpt --results_dir ./results --embedding shuffle_PreAct --dist G --num_workers 40

\time python Mulit_task_SSL/main2.py