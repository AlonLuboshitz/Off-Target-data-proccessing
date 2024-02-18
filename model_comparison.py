## THIS SCRIPT CREATES INPUTS,OUTPUTS FOR OTHER STATE OF THE ART MODELS IN ORDER TO COMPARE THEIR RESULTS

#### DATA
'''From set of guides given, and data frame of guides, return the data frame of the guides that are in the set'''
def get_guides_data(guides_set, data_frame, target_column):
    return data_frame[data_frame[target_column].isin(guides_set)]


#### MODELS

## CRISPRNET

''' Create input.txt to CRISPRNET model:
    Input file should be gRNA ('on_seq'), Off-target sequence ('off_seq')'''
def create_input_crisprnet(data_frame, target_column, off_target_column , output_path):
    data_frame[[target_column, off_target_column]].to_csv(output_path, index=False, header=False, sep=",")
import pandas as pd
from constants import TARGET_COLUMN, OFFTARGET_COLUMN, CHANGESEQ_5k_TEST, CHANGESEQ_GS_EPI
''' Iterate on test guides and create input files for CRISPRNET model'''
def create_crisprnet_input_files(test_guides_path, data_frame_path, target_column, off_target_column, output_folder):
    test_guide = pd.read_csv(test_guides_path)
    data_frame= pd.read_csv(data_frame_path)
    for i in range(len(test_guides_path)):
        guides = test_guide.loc[i,target_column]
        
        create_input_crisprnet(get_guides_data(guides, data_frame, target_column), target_column, off_target_column, output_folder + f"/input_{i}.txt")

create_crisprnet_input_files(CHANGESEQ_5k_TEST, CHANGESEQ_GS_EPI, TARGET_COLUMN, OFFTARGET_COLUMN, "/home/alon/masterfiles/pythonscripts/Changeseq/CRISPRNET/inputs")