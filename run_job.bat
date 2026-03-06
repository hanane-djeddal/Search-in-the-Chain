#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=searchchain #segmentationMixedCum # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds
#SBATCH --gpus=1 # nombre de gpu ##SBATCH --nodelist=zz
#SBATCH --mem-per-gpu=80G
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=48:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
#conda activate RagnRoll
#conda activate llm-agent
conda activate selfragenv
cd /home/djeddal/Documents/Code/Search-in-the-Chain


python SearChain_w_IR.py --dataset asqa --file /home/djeddal/Documents/Code/CORAL/coral_retrieval_subq.json