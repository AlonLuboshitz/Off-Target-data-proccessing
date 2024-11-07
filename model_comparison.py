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
    4. output_path: path to save the input file to the model
    
    create_input_moff_score(original_data_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",
                         target_column="target", off_target_column="offtarget_sequence",
                         output_path = External_models_folder, data_name="Hendel_only_mism")'''
    original_data = pd.read_csv(original_data_path)
    data = original_data[[target_column, off_target_column]]
    full_output_path = f'{output_path}/MOFF/{data_name}'
    # remove OTS with indels ("-") or letters other than A, T, C, G
    filtered_data, removed_indices =  remove_unwanted_otss('ATGC', data, off_target_column,full_output_path)
    full_output_path = full_output_path + "_input_toMOFF_score.csv"
    filtered_data.to_csv(full_output_path, header=False, index=False)

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
        indices_path = indices_path + "_removed_indices.csv"
        print(f"Saving removed indices to {indices_path}")
    else: 
        print("Save indices in working folder")
        indices_path = "removed_indices.csv"
    removed_indices.to_csv(indices_path, index=False, header = False)
    return data_cleaned, removed_indices

def complete_missing_values(model_output_path, indices_path, column_to_complete, averaged_scores = None, default_value = 0.5):
    '''This function complete missing values for OTSs that the model cant except.
    For example, MOFF cant except OTSs with indels or other codings than ATGC.
    For this OTSs, we will set a defualt value of 0.5 or run the model on the optional codings, i.e. for OT = NNN...Y..NN
    we will run the model on all possible codings of Y and take the average of the predictions.
    
    Args:
    1. model_output_path: path to the model output file
    2. indices_path: path to the indices file of the OTSs that were removed
    3. column_to_complete: column name of the column to complete
    4. averaged_scores: list of averaged scores for the OTSs that were removed corresponding to the indices
    5. default_value: default value to set for the missing values
    
    complete_missing_values("/home/dsi/lubosha/Off-Target-data-proccessing/External_models_predictions/MOFF/MOFF_scores/hendel_only_mism_MOFF.score.csv", 
                        "/home/dsi/lubosha/Off-Target-data-proccessing/External_models_predictions/MOFF/Hendel_only_mism_removed_indices.csv",None)

    '''

    
    model_output = pd.read_csv(model_output_path)  # Load the model output
    indices = pd.read_csv(indices_path, header=None).values.flatten() # Load indices of removed OTSs
    indices = sorted(indices)  # Sort indices to process them in order
    
    # Initialize an empty list to collect parts of the DataFrame
    parts = []
    start = 0  # Start position for splitting
    for j,index in enumerate(indices):
        parts.append(model_output.iloc[start:index])  # Append the part of the DataFrame before the current index
        index_df = pd.DataFrame(columns=model_output.columns)  # Create a DataFrame with the same columns as the model output
        index_df.loc[0] = default_value if averaged_scores is None else averaged_scores[j]  # Set the default value
        parts.append(index_df)  # Append the DataFrame with the default value
        start = index 
    # fill rest of the columns
    parts.append(model_output.iloc[start:])
    filled_model_output = pd.concat(parts, ignore_index=True)  # Concatenate the parts into a new DataFrame
    new_path = model_output_path.split(".")[0] + "_completed.csv"
    filled_model_output.to_csv(new_path, index=False)

def combine_external_model_outputs(model_output_path, original_data_path, model_output_columns):
    '''This function combines the external model outputs into the original OTS data frame (original data).
    It assumes that the model output is in the same order as the original data and that the number of points is the same.
    
    Args:
    1. model_output_path: path to the model output scores
    2. original_data: data frame with the original OTS data
    3. model_output_columns: list of columns need to be added to the original data frame

    combine_external_model_outputs("/home/dsi/lubosha/Off-Target-data-proccessing/External_models_predictions/MOFF/MOFF_scores/hendel_only_mism_MOFF_completed.csv",
                               "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",
                               ["GMT","MOFF"])'''
    
    model_scores = pd.read_csv(model_output_path)
    original_data = pd.read_csv(original_data_path)
    if len(original_data) != len(model_scores):
        raise ValueError("The number of points in the model output and the original data is not the same.")
    for column in model_output_columns:
        if column in original_data.columns:
            print(f"Model column: {column} already in the original data frame, WILL OVERIDE EXSITING MODEL DATA IN ORIGINAL DATA") 
    original_data[model_output_columns] = model_scores[model_output_columns] # Add the model output to the original data frame
    new_output = original_data_path.split(".")[0] + "_with_model_scores.csv"
    original_data.to_csv(new_output, index=False)  # Save the updated data frame

