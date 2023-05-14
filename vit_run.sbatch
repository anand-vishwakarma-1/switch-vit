#!/bin/bash
#SBATCH --job-name=vit_model_rtx_4_dmodel_300_experts_32_batch_256_cifar100_cosine
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --time=15:00:00
#SBATCH --output="vit_model_rtx_4_dmodel_300_experts_32_batch_256_cifar100_cosine.txt"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=asv8775@nyu.edu


module purge

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT

export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

singularity \
  exec --nv \
  --overlay /scratch/asv8775/overlays/hpml-project.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
 source /ext3/env.sh
 mkdir /scratch/asv8775/hpml/project/models/vit_model_rtx_4_dmodel_300_experts_32_batch_256_cifar100_cosine
 export PYTHONUNBUFFERED=TRUE
 python vit.py --batch 256 --experts 32 --epochs 500 --gpu 4 --dmodel 300 --cifar100 --out vit_model_rtx_4_dmodel_300_experts_32_batch_256_cifar100_cosine
"