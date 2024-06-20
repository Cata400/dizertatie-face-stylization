#!/bin/bash

declare -A styles
styles=(
    ["celeba"]="/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/"
    ["aahq"]="/home/catalin/Desktop/Disertatie/Datasets/aahq/aligned_used/"
    ["sketches"]="/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/"
)

results_dir="../Results/"
result_file="../Metrics/fid_results.txt"

# # Activate conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate fid

# Clear the result file
: > "$result_file"

for directory in $(ls "$results_dir" | sort); do
    if [[ "$directory" == *"ffhq"* && "$directory" != *"old"* ]]; then
        style=$(echo "$directory" | cut -d'_' -f3)
        
        echo "Calculating FID for $directory with ${styles[$style]} ..."
        echo "Calculating FID for $directory with ${styles[$style]} ..." >> "$result_file"
        
        command="python -m pytorch_fid ${styles[$style]} ${results_dir}${directory} --device cuda:0"
        $command >> "$result_file"
        
        echo ""
        echo "" >> "$result_file"
    fi
done