#!/bin/bash
#SBATCH --job-name=a100_hagrid_searchinchain ###contrastivesearch_p4 ### diversity_sampling_querycontinue
#SBATCH --output=jz_%j_%x.out    # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out  
#SBATCH --gres=gpu:8   #for v account max is 4  #SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=30
#SBATCH --account=fiz@a100
#SBATCH -C a100

module purge
module load arch/a100
module load pytorch-gpu/py3/2.0.1
module purge
module load arch/a100
module load pytorch-gpu/py3/2.0.1

export JAVA_HOME=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export JVM_PATH=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2/lib/server/libjvm.so
export PYSERINI_CACHE=$WORK/.cache/pyserini
export HF_HOME=$WORK/.cache/huggingface/
export PYSERINI_CACHE=$WORK/.cache/pyserini

cd /lustre/fswork/projects/rech/fiz/udo61qq/Code/Search-in-the-Chain/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python SearChain_w_IR.py 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python SearChain_w_IR.py --dataset  asqa
