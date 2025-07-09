#!/bin/bash
#SBATCH --job-name=generate_noises         # Job name
#SBATCH --output=dmsr_%j.out         # Standard output and error log
#SBATCH --error=dmsr_%j.err          # Separate error log (optional)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=16                 # CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=02:00:00                   # Max runtime hh:mm:ss
#SBATCH --partition=gpu                   # GPU partition (change as needed)

unset SLURM_EXPORT_ENV
# Define project directory on the cluster
PROJECT_DIR=/project/DMSR

# Build the Apptainer container from the Docker image
apptainer build dmsr.sif ${PROJECT_DIR}/Dockerfile
# Run the Python script inside the container, bind mounting your project directory
apptainer exec --nv --bind ${PROJECT_DIR}:/project --nv --bind /hnvme/workspace/b266be10-storage_tom/:/workspace/tom_storage/  dmsr.sif python /project/generate_noisy_set.py




