## THIS SCRIPT CREATES INPUTS,OUTPUTS FOR OTHER STATE OF THE ART MODELS IN ORDER TO:
# 1. Compare their prediction performance results with the proposed model
# 2. Use their predictions as features for the proposed model

'''The script contains the same functions for each model:
1. Create input files for the model
2. Run the model and create output files'''
import pandas as pd


## GLOBALS ##
External_models_folder = "/home/dsi/lubosha/Off-Target-data-proccessing/External_models_predictions"
#### DATA
'''From set of guides given, and data frame of guides, return the data frame of the guides that are in the set'''
def get_guides_data(guides_set, data_frame, target_column):
    return data_frame[data_frame[target_column].isin(guides_set)]

#### INPUTS
def create_input(original_data_path, target_column, off_target_column, model_name, output_path, data_name):
    '''Create input file for the model'''
    if model_name == "CRISPRNET":
        create_input_crisprnet(original_data_path, target_column, off_target_column, output_path, data_name)
    elif model_name == "MOFF":
        print("Create input for MOFF")
        create_input_moff_score(original_data_path, target_column, off_target_column, output_path, data_name)
    else:
        raise ValueError("Model name not recognized")

#### MODELS

## CRISPRNET


def create_input_crisprnet(data_frame, target_column, off_target_column , output_path):
    ''' Create input.txt to CRISPRNET model:
    Input file should be gRNA ('on_seq'), Off-target sequence ('off_seq')'''
    data_frame[[target_column, off_target_column]].to_csv(output_path, index=False, header=False, sep=",")

def create_crisprnet_input_files(test_guides_path, data_frame_path, target_column, off_target_column, output_folder):
    ''' Iterate on test guides and create input files for CRISPRNET model'''
    test_guide = pd.read_csv(test_guides_path)
    data_frame= pd.read_csv(data_frame_path)
    for i in range(len(test_guides_path)):
        guides = test_guide.loc[i,target_column]
        create_input_crisprnet(get_guides_data(guides, data_frame, target_column), target_column, off_target_column, output_folder + f"/input_{i}.txt")

## MOFF

def create_input_moff_score(original_data_path, target_column, off_target_column, output_path, data_name):
    '''MOFFscore requires the user to provide .csv or .txt file containing sgRNA sequences and corresponding DNA target sequences.
    Each line should have one gRNA(20bp+PAM) and one target(20bp+PAM) sequence.
    Note that MOFF is designed for mismatch-only off-target prediction, not for indel mutations
    Example: sgRNA, Off-target
    
    The function creates the input file for MOFF by taking the original data frame and extracting the target and off-target columns
    Args:
    1. original_data: data frame with the original data
    2. target_column: column name of the target
    3. off_target_column: column name of the off-target
    4. output_path: path to save the input file to the model'''
    original_data = pd.read_csv(original_data_path)
    data = original_data[[target_column, off_target_column]]
    full_output_path = f'{output_path}/MOFF/{data_name}'
    # remove OTS with indels ("-") or letters other than A, T, C, G
    filtered_data, removed_indices =  remove_unwanted_otss('ATGC', data, off_target_column,full_output_path)
    full_output_path = full_output_path + "_input_toMOFF_score.txt"
    filtered_data.to_csv(full_output_path, header=False, index=False, sep="\t")

def remove_unwanted_otss(valid_coding, data, off_target_column, indices_path=None):
    '''This functions removes off-targets that contain unwanted codings from the data frame.
    Args: 1. valid_coding: list of codings that can be kept, all other will be removed.
          2. data: data frame with the off-targets
          3. off_target_column: column name of the off-targets
          4. indices_path: path to save the indices of the removed off-targets
    Returns: data frame without the unwanted codings, and the indices of the removed off-targets'''
    valid_pattern = f'^[{valid_coding}]+$'  # Pattern to match strings composed only of valid codings
    # Identify indices of sequences that do NOT match the valid pattern
    removed_indices = pd.Series(data.index[~data[off_target_column].str.match(valid_pattern)])
    # Filter out invalid sequences
    data_cleaned = data[data[off_target_column].str.match(valid_pattern)]
    print(f"Removed : {len(removed_indices)} of invalid OTSs that contains other coding than: {valid_coding}")
    if indices_path:
        indices_path = indices_path + "_removed_indices.txt"
        print(f"Saving removed indices to {indices_path}")
    else: 
        print("Save indices in working folder")
        indices_path = "removed_indices.txt"
    removed_indices.to_csv(indices_path, index=False, header = False, sep="\t")
    return data_cleaned, removed_indices


    

create_input_moff_score(original_data_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",
                        target_column="target", off_target_column="offtarget_sequence",
                        output_path = External_models_folder, data_name="Hendel_only_mism")