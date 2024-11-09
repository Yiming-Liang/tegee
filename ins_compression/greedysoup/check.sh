#!/bin/bash

# Check if directory argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_path> <max_task_number>"
    exit 1
fi

# Get directory path from command-line argument
DIR="$1"

# Get maximum task number (N) from command-line argument
N="$2"

# Initialize counter for missing folders
missing_count=0

# Loop over each expected folder name
for i in $(seq 0 $N); do
    folder_name="${i}"
    folder_path="${DIR}/${folder_name}"

    # Check if the folder does not exist
    if [[ ! -d "${folder_path}" ]]; then
        echo "Missing: ${folder_name}"
        ((missing_count++))  # Increment missing folder count
    fi
done

# Output the total number of missing folders
echo "Total number of missing folders: ${missing_count}"

