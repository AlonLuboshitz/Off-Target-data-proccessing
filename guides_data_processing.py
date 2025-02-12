'''
This script is used to create or update a csv file that holds guide information and its off target statistics.

The csv file has the following structure:
Header: Data_set, Year, Gene_name, guide_sequence, amplified_otss, amplified_method, genome_wide_otss, genome_wide_method, potential_otss, potential_method, genome_amplified_intersect_otss, notes
In the otss methods the value will be method, cell checked.

The following functions are aviailable:'''
# Globals
PATH_TO_STATISTICS_DATA = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/guides_statistics.csv'
HEADER = ['Data_set', 'Year', 'Gene_name', 'guide_sequence', 'amplified_otss', 'amplified_method', 'vivo_otss', 'vivo_method', 'vitro_otss','vitro_method','potential_otss', 'potential_method', 'genome_amplified_intersect_otss', 'notes']
import pandas as pd
import os

def create_guides_statistics_file():
    '''
    This function creates a csv file that holds guide information and its off target statistics.
    '''
    if os.path.exists(PATH_TO_STATISTICS_DATA):
        print('File already exists')
        return
    with open(PATH_TO_STATISTICS_DATA, 'w') as f:
        f.write(','.join(HEADER) + '\n')
    print('File created')
def open_data():
    '''
    This function opens the csv file that holds guide information and its off target statistics.
    '''
    if not os.path.exists(PATH_TO_STATISTICS_DATA):
        create_guides_statistics_file()
    return pd.read_csv(PATH_TO_STATISTICS_DATA)
def input_data_to_statistic_file(data, active_label, treshold, method, guide_column, data_name=None,
                                 cell= None, year=None, gene_name=None, ):
    '''
    This function inputs data to the csv file that holds guide information and its off target statistics.
    Args:
    1. data - (data_frame/path) to off target data.
    2. active_label - (str) the active column for OT.
    3. treshold - (int) the treshold for the active column. above that treshold off target is considered active.
    4. method - (str) the method used to obtain the OT.
    '''
    if isinstance(data, str):
        data = pd.read_csv(data)
    guides = data[guide_column].unique()

def input_to_data_by_column(data_name, guides_list, values, column_to_fill):
    """
    Function to group data by data_name and fill corresponding columns based on guides and values.

    Args:
        data (pd.DataFrame): Existing dataframe to update. Can be empty.
        data_name (str): Value to go into 'Data_set' column.
        guides_list (list): List of guides to populate 'guide_sequence' column.
        values_dict (list): list of tuple with values corresponds to the columns to fill.
        columns_to_fill (list): Names of the columns to fill with values.
        header (list): List of column names for the dataframe.

    Returns:
        pd.DataFrame: Updated dataframe with new rows/values.
    """
    # Ensure the dataframe has the correct structure
    
    data = open_data()
    # Check if data_name exists in the dataframe
    if data_name not in data['Data_set'].values:
        # Add a new group for the data_name
        for guide in guides_list:
            new_row = {col: None for col in HEADER}  # Initialize a new row with None values
            new_row['Data_set'] = data_name
            new_row['guide_sequence'] = guide
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    # Iterate over guides and fill columns
    for guide, value in zip(guides_list, values):
    
        # Locate the row(s) corresponding to the guide and data_name
        mask = (data['Data_set'] == data_name) & (data['guide_sequence'] == guide)
        if not mask.any():
            # If no row exists for the guide, add it
            new_row = {col: None for col in HEADER}
            new_row['Data_set'] = data_name
            new_row['guide_sequence'] = guide
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
            mask = (data['Data_set'] == data_name) & (data['guide_sequence'] == guide)

        # Fill the specified columns with values
        data.loc[mask, column_to_fill] = value
    data.to_csv(PATH_TO_STATISTICS_DATA, index=False)
#genes = [tuple(x) for x in data[['sample', 'another_column', 'third_column']].values]

def sum_target_otss(data, target, label):
    '''Return active otss for data grouped by target'''
    groups = data.groupby(by=target)
    guides = data[target].unique()
    active_otss = []
    for guide in guides:
        group = groups.get_group(guide)
        active_otss.append(sum(group[label]> 0))
    print(sum(active_otss))
    return guides, active_otss

def potenital_ots(data,target,guide_=None):
    # vivo+silico/vitro+silico
    '''Return potential otss for data grouped by target
    if guide_ is given check only this guides'''
    groups = data.groupby(by=target)
    guides = data[target].unique()
    if guide_:
        guides = guide_
    potential_otss = []
    for guide in guides:
        group = groups.get_group(guide)
        potential_otss.append(len(group))
    return guides,potential_otss  


#HEADER = ['Data_set', 'Year', 'Gene_name', 'guide_sequence', 'amplified_otss', 'amplified_method', 'vivo_otss', 'vivo_method', 'vitro_otss','vitro_method','potential_otss', 'potential_method', 'genome_amplified_intersect_otss', 'notes']
