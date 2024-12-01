#!/bin/bash

# File containing the list of used nodes
USED_NODES_FILE="/home/hfontaine/MAX78000_face_detection/used_nodes.txt"

# Check if the file exists
if [[ ! -f "$USED_NODES_FILE" ]]; then
    echo "Error: File $USED_NODES_FILE does not exist."
    exit 1
fi

# Read nodes from the file
USED_NODES=$(cat "$USED_NODES_FILE")

# Loop over each node and submit a cleanup job
for NODE in $USED_NODES; do
    echo "Submitting cleanup job for node: $NODE"

    # SLURM job submission for cleanup
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cleanup_$NODE
#SBATCH --output=/home/hfontaine/MAX78000_face_detection/cleanup_logs/cleanup_${NODE}_%j.txt
#SBATCH --error=/home/hfontaine/MAX78000_face_detection/cleanup_logs/cleanup_${NODE}_%j.err
#SBATCH --nodelist=$NODE
#SBATCH --mail-type=NONE

echo "Cleaning up on node: $NODE"

# Remove local conda environment directory
rm -rf /scratch/$USER/conda_env || echo "Failed to remove /scratch/$USER/conda_env on $NODE"
EOF

done

# Confirm cleanup jobs were submitted
echo "All cleanup jobs have been submitted."