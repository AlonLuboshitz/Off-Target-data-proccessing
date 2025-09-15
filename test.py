import numpy as np
import pickle
import os
from utilities import extract_scores_labels_indexes_from_files
import os
import glob

root_dir = "/localdata/alon/ML_results/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/10_ensembels"

# Iterate over all folders in root_dir
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        scores_path = os.path.join(folder_path, "Scores")
        print('Processing folder: ',scores_path)
        if os.path.isdir(scores_path):
            # Find all ensemble_*.csv files in Scores folder
            pattern = os.path.join(scores_path, "ensemble_*.csv")
            for csv_file in glob.glob(pattern):
                print(f"Processing file: {csv_file}")
                scores, labels, indexes = extract_scores_labels_indexes_from_files([csv_file])
                scores = np.mean(scores,axis=0)
                csv_file = csv_file.replace('.csv','.pkl')
                with open(csv_file,'wb') as f:
                    pickle.dump(scores,f)
# scores, labels, indexes = extract_scores_labels_indexes_from_files([path_1])

# # score_files = [os.path.join(path, file) for file in os.listdir(path) ]
# # for file in score_files:
    

# scores_,labels_, indexes_ = extract_scores_labels_indexes_from_files([path_2])
# if np.array_equal(indexes, indexes_):
#     print("Indexes are equal")
# if np.array_equal(labels, labels_):
#     print("Labels are equal")