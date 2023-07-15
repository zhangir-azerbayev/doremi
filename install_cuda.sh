#!/bin/bash
#SBATCH --job-name="llmath_install_cuda"
# #SBATCH --account=dw87
#SBATCH --comment="eleutherai"
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=install_cuda_%j.out
#SBATCH --error=install_cuda_%j.out
#SBATCH --time=3-00:00:00

# Designed for compute nodes that don't have internet. 
# Must predownload conda packages using conda install --download-only. 
# Must also populate doremi with all other required packages.
source /home/hailey81/miniconda3/bin/activate doremi 

echo $CONDA_PREFIX

export CONDA_OFFLINE=true

conda install -c conda-forge cudatoolkit=11.7 gxx=10.3.0 pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

DOREMI_DIR=/home/za2514/compute/doremi

cd ${DOREMI_DIR}/flash-attention && python setup.py install
cd ${DOREMI_DIR}/flash-attention/csrc/fused_dense_lib && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/xentropy && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/rotary && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/layer_norm && pip install .

echo "exited succesfully"
