#!/bin/bash
#SBATCH -J jax-psum
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --image=ghcr.io/nvidia/jax:jax-2024-09-08
#SBATCH --module=gpu,nccl-plugin

export MASTER_ADDR=$(hostname)
#export FI_LOG_LEVEL=trace
export NCCL_DEBUG=INFO

# Manual setup of GDR, can be done before running job
mkdir -p $SCRATCH/shifter-gdr
rsync -vau /usr/lib64/libgdrapi.so* $SCRATCH/shifter-gdr/

srun -u -l shifter bash -c "
    export LD_LIBRARY_PATH=$SCRATCH/shifter-gdr:\$LD_LIBRARY_PATH
    python jax_psum.py
"
