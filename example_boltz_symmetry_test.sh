#!/bin/bash
#SBATCH -A mrsec
#SBATCH -p mrsec-gpu-long
#SBATCH --qos=medium
#SBATCH --gres=gpu:TitanXP:1
#SBATCH --job-name=boltz_symmetry_test
#SBATCH --time=7-00:00:00

# Load necessary modules (if applicable)
# module load cuda/XX.X  # Uncomment and specify if required

# Activate the Conda environment
source activate /work/keshavsundar/env/boltz

# Run the prediction command
boltz predict test.yaml --pdb updated3_updated_AD_C3_11275_I1_D4_I6_D1_I28_D4_rotated_51_degrees_IC3_final_10A_redesigned_partial_diffusion_setup.pdb  --cache /work/keshavsundar/env/boltz/weights
