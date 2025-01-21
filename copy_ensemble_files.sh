#!/bin/bash

# Define base paths
base_10="/localdata/alon/ML_results/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/10_ensembels/50_models/Binary_epigenetics"
base_1="/localdata/alon/ML_results/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics"

# Loop through subdirectories in 1_ensembles
for dir in "$base_1"/*/Scores; do
    # Extract subdirectory name
    subdir=$(basename "$(dirname "$dir")")
    
    # Define source and target paths
    src="$dir/ensemble_1.csv"
    dest="$base_10/$subdir/Scores/"
    
    # Check if source file exists
    if [[ -f "$src" ]]; then
        # Create target directory if it doesn't exist
        mkdir -p "$dest"
        
        # Copy the file
        cp "$src" "$dest"
        echo "Copied $src to $dest"
    else
        echo "Source file $src does not exist. Skipping."
    fi
done
