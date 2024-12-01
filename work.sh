#!/bin/bash
# SBATCH directives must come before the first executable line

#SBATCH --mail-type=NONE                                                            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/hfontaine/MAX78000_face_detection/results/results_%j.txt     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/hfontaine/MAX78000_face_detection/results/errors_%j.txt       # where to store error messages
#SBATCH --gres=gpu:1                                                                # Request 1 GPU

# Uncomment the line below to request a specific node, or leave it commented to run on any node
#SBATCH --nodelist=artongpu01                                                       # Request a specific node


echo "Starting script..."

# ==========================
# User-defined parameters
# ==========================
# Define path without trailing slash
USED_NODES_FILE="/home/hfontaine/MAX78000_face_detection/used_nodes.txt"
PYTHON_SCRIPT_PATH="/home/hfontaine/MAX78000_face_detection/work.py"
LOCAL_CONDA_ENV_PATH="/itet-stor/hfontaine/net_scratch/venv/conda_envs/cluster"
DATASET_SOURCE="/itet-stor/hfontaine/net_scratch/datasets/widerface"
NODE_PERSONAL_DIRECTORY="/scratch/$USER"

# ==========================
# Script logic
# ==========================
NODE_CONDA_ENV="$NODE_PERSONAL_DIRECTORY/conda_env"
PYTHON_INTERPRETER="$NODE_PERSONAL_DIRECTORY/conda_env/bin/python3"
NODE_DATASET_DIRECTORY="$NODE_PERSONAL_DIRECTORY/datasets"

# Function to clean up the environment and temporary files
cleanup() {
    if [[ $CLEANUP_FLAG == "1" ]]; then
        echo "Cleaning up temporary files, conda environment, and datasets..."
        rm -rf "$TMPDIR"
        rm -rf "$NODE_CONDA_ENV"
        rm -rf "$NODE_DATASET_DIRECTORY"
        echo "Cleanup complete."
    else
        echo "No cleanup required."
    fi
}

# Trap to call cleanup on exit
trap cleanup EXIT

# Exit on errors
set -o errexit

# Flag to indicate if cleanup is necessary
CLEANUP_FLAG="1"

# Set a directory for temporary files unique to the job
TMPDIR=$(mktemp -d)
if [[ ! -d "$TMPDIR" ]]; then
    echo "Failed to create temp directory" >&2
    exit 1
fi
export TMPDIR

# Change the current directory to the temporary directory
cd "$TMPDIR" || {
    echo "Failed to change directory to TMPDIR" >&2
    exit 1
}

# Log noteworthy information
NODE=$(hostname)
echo "Running on node: $NODE"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Log the current node to the used nodes file
{
    echo "$NODE" >> "$USED_NODES_FILE"
} || {
    echo "Failed to write to $USED_NODES_FILE" >&2
}

# Ensure no duplicates in the used nodes file
sort -u "$USED_NODES_FILE" -o "$USED_NODES_FILE"

# Create a local scratch directory
mkdir -p "$NODE_PERSONAL_DIRECTORY" || {
    echo "Failed to create local scratch directory" >&2
    exit 1
}

# Copy the virtual environment to the local scratch
# /!\ remove --ignore-existing to update the environment
rsync -avq --ignore-existing "$LOCAL_CONDA_ENV_PATH/" "$NODE_CONDA_ENV/" --compress || {
    echo "Failed to copy conda environment" >&2
    exit 1
}

# Copy the dataset to the local scratch
# /!\ remove --ignore-existing to update the dataset
rsync -avq --ignore-existing "$DATASET_SOURCE" "$NODE_DATASET_DIRECTORY/" --compress || {
    echo "Failed to copy dataset" >&2
    exit 1
}

# Check if the Python interpreter exists
if [[ ! -x "$PYTHON_INTERPRETER" ]]; then
    echo "Python interpreter not found at $PYTHON_INTERPRETER" >&2
    exit 1
fi

# If all setup steps succeeded, disable cleanup
CLEANUP_FLAG="0"

# Execute the Python script using the copied environment's interpreter
"$PYTHON_INTERPRETER" "$PYTHON_SCRIPT_PATH" || {
    echo "Python script execution failed" >&2
    exit 1
}

# Log completion
echo "Finished at:     $(date)"

# Exit with success code
exit 0
