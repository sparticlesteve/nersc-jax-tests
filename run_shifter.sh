#!/bin/bash
#SBATCH -J jax-psum
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --image=ghcr.io/nvidia/jax:jax-2024-09-08
#SBATCH --module=gpu,nccl-plugin

export MASTER_ADDR=$(hostname)

srun -u -l shifter python jax_psum.py
