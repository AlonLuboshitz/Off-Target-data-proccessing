import numpy as np
import os
import re
from itertools import combinations
from shutil import rmtree
import pandas as pd
import ast



### DATA UTILITIES ###

def keep_positives_by_ratio(X_data, y_data, ratio = None):
    '''This function keeps a subset ratio of the positive data points in the data.
    If ratio is None/<=0/>1 it will through an error.
    Other wise it will keep the ratio of the positive data points for each guide!
    For example if ratio = 0.5 it will keep half of the positive data points for each guide keeping the guide amount the same!
    Args:
    1. X_data - data points - [n_guides,np.array(datapoints,one_hot_vectors)]
    2. y_data - labels - [n_guides,np.array(datapoints,labels)]
    3. ratio - ratio of the positive data points to keep.
    -----------
    Returns: X-data*ratio and y_data*ratio'''
    if ratio is None or ratio <= 0 or ratio >1:
        raise Exception("Ratio must be a number between 0 and 1")
    
    # Sort the data by the labels
    sorted_indices = [] 
    X_data_copy = X_data.copy()
    y_data_copy = y_data.copy()
    for index,labels_array in enumerate(y_data_copy):
        labels_array = labels_array.ravel() # flatten the array
        positive_indices = np.where(labels_array > 0)[0] # keep only the positive labels
        total_amount = len(positive_indices)
        sorted_indices = np.argsort(labels_array[positive_indices])[::-1] # get the indices that sort the labels
        amount_to_keep = int(total_amount*ratio) # amount of positive data points to keep
        if amount_to_keep <= 2 or total_amount <= 2: # keep at least 2 positive data points
            sorted_indices = sorted_indices[:2]
            y_data_copy[index] = y_data_copy[index][sorted_indices] 
            X_data_copy[index] = X_data_copy[index][sorted_indices] 
            continue
        sorted_indices = sorted_indices[:amount_to_keep] # keep the first ratio of the positive data points
        y_data_copy[index] = y_data_copy[index][sorted_indices] # sort the labels
        X_data_copy[index] = X_data_copy[index][sorted_indices] # sort the features
        if len(y_data_copy[index]) != len(X_data_copy[index]):
            raise Exception("The labels and features must have the same amount of data points")
        
    return X_data_copy,y_data_copy

def keep_negatives_by_ratio(X_data, y_data, ratio = None):
    '''This function keeps a subset ratio of the negative data points in the data.
    If ratio is None/<=0/>1 it will through an error.
    Other wise it will keep the ratio of the negative data points for each guide!
    For example if ratio = 0.5 it will keep half of the negative data points for each guide keeping the guide amount the same!
    Args:
    1. X_data - data points - [n_guides,np.array(datapoints,one_hot_vectors)]
    2. y_data - labels - [n_guides,np.array(datapoints,labels)]
    3. ratio - ratio of the negative data points to keep.
    -----------
    Returns: X-data*ratio and y_data*ratio'''
    if ratio is None or ratio <= 0 or ratio >1:
        raise Exception("Ratio must be a number between 0 and 1")
   
    X_data_copy = X_data.copy()
    y_data_copy = y_data.copy()
    for index,labels_array in enumerate(y_data_copy):
        labels_array = labels_array.ravel() # flatten the array
        negative_indice = np.where(labels_array==0)[0] # keep only the negative labels indices
        total_amount = len(negative_indice)
        amount_to_keep = int(total_amount*ratio) # amount of positive data points to keep
        
        if amount_to_keep < 2: # keep at least 2 positive data points
            continue
        shuffled_indices = np.random.permutation(negative_indice)[:amount_to_keep] # shuffle the indices and keep the first ratio of the negative data points
        y_data_copy[index] = y_data_copy[index][shuffled_indices] # sort the labels
        X_data_copy[index] = X_data_copy[index][shuffled_indices] # sort the features
        if len(y_data_copy[index]) != len(X_data_copy[index]):
            raise Exception("The labels and features must have the same amount of data points")
    return X_data_copy,y_data_copy

def keep_positive_OTSs_labels(y_scores, y_test, indexes):
    '''Function keeps only the positive OTSs and their corresponding scores.
    It remove the zero labels to avoid floating point errors.
    Args:
    1. y_test - true labels
    2. y_scores - predicted labels
    3. indexes - indexes of the data points
    -----------
    Returns: positive OTSs labels, scores and indexes'''
    zero_y_test = np.where(y_test == 0)[0] # zero indices
    pos_y_test = np.delete(y_test,zero_y_test) # get only the positive OTSs
    
    pos_y_scores = np.delete(y_scores,axis=1,obj=zero_y_test) # get only the predicted positive OTSs
    pos_indexes = np.delete(indexes,zero_y_test) # get only the positive OTSs indexes
    return pos_y_scores, pos_y_test, pos_indexes
def convert_partition_str_to_list(partition_str):
    '''This function will convert a partition string to a list of integers
    Args:
        partition_str - string with the partition
        -----------
        returns a list of integers
        '''
    partitions_nums = partition_str.split("_")[0]
    num_strings = partitions_nums.split('-')
    return [int(num) for num in num_strings]

def validate_dictionary_input(answer, dictionary):
    '''This function is used to validate the input from the user.
 It checks if the input is a valid number and if it is within the range of the list of options
 given by keys of the dictionary'''
    if not answer: # no input was given
        answer = input(f"Please enter model: {dictionary}\n") # ask for model
    # input was given
    answer = int(answer)
    if answer not in dictionary.keys(): # check if the input is in the dictionary
        raise Exception(f"Invalid input. Please choose from {dictionary.keys()}")
    else :
        return answer


def create_guides_list(guides_path,i_line):
    '''function path to guides txt file and return a list from the i line of the file
    i_line is the line number to read from the file
    the returned list objects are gRNA strings separted by comma "," '''
    with open(guides_path, "r") as f:
        for i, line in enumerate(f):
            if i == i_line:
                line = line.replace("\n","")
                line2 = line.split(",")
                guides = [guide.replace(' ','') for guide in line2]
                break
    return guides
def extract_guides_from_partition(partition_info,partition):
    '''Given partition information and the partition number extract the guides of the partition'''
    if not isinstance(partition_info,pd.DataFrame):
        info = pd.read_csv(partition_info)
    else: info = partition_info
    partition_info_guides = info[info["Partition"] == partition]["Guides"].values[0]
    try:
        partition_guides = ast.literal_eval(partition_info_guides)
        
    except Exception as e:
        print(f"Error: {e}")
        return e
    return partition_guides  
        
'''Function writes 2d array to csv file'''
def write_2d_array_to_csv(np_array, file_path, header):
    if np_array.ndim != 2:
        raise Exception("np_array must be 2d")
    if file_path.split(".")[-1] != "csv":
        raise Exception("file_name must end with csv")
    if header: # not None/empty
        if len(header) != np_array.shape[1]:
            raise Exception("header must be the same length as the number of columns in the np_array") 
   
    np.savetxt(file_path, np_array, delimiter=',', fmt='%.5f', header=','.join(header), comments='')
def add_row_to_np_array(y_scores, y_test):
    # if y_scores.dtype != y_test.dtype:
    #     raise Exception("y_scores and y_test must have the same dtype")
    if y_scores.ndim != 2 or y_test.ndim != 1:
        raise Exception("y_scores must be 2d and y_test must be 1d")
    if y_scores.shape[1] != y_test.shape[0]:
        raise Exception("y_scores must have the same number of columns as y_test number of values")
    return np.vstack((y_scores, y_test))
    
def split_epigenetic_features_into_groups(features_columns):
    '''Give a list of features combine the features into groups based on
      their endings - binary, score, enrichment, etc.
      
      Returns a dictionary with the groups as keys and the features as values.'''

    # Create a dictionary to store groups based on endings
    groups = {}
    # Group strings based on their endings
    for feature in features_columns:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups  

def set_features_columns_by_string( feature_string, split_by,epi_feature_columns = None,other_columns = None,method = None):
    '''This function take a string of epigentic features and return a list of the features matching the
the features columns
features_columns - name of the epigenetic features columns
epi_string - string of epigenetic features
split_by - how to split the epi_string
--------
returns a list of the features matching the epi_string'''
    if method == 2: # epi_features
        if epi_feature_columns is None:
            raise Exception("epi_feature_columns  must be given")
        feature_columns = epi_feature_columns
    elif method == 5: # other features
        if other_columns is None:
            raise Exception("other_columns  must be given")
        feature_columns = other_columns
    elif method == 6: # all features
        if epi_feature_columns is None or other_columns is None:
            raise Exception("epi_feature_columns and other_columns  must be given")
        feature_columns = epi_feature_columns + other_columns
    if feature_string == "All":
        return feature_columns
    epi_features = feature_string.split(split_by)
    return_list = []
    for feature in epi_features:
        feature = feature.strip().lower()  # Remove whitespace and convert to lowercase
        if feature == "chromstate" or feature == "binary" or feature == "peaks":
            continue
        for column in feature_columns:
            if feature in column.strip().lower():
                return_list.append(column)
    return return_list
    
def get_feature_name(feature_column):
    '''This function splits the feature column by _ and returns the name of the feature'''
    feature_column_splited = feature_column.split("_")
    if feature_column_splited[0] == 'Chromstate':
        return feature_column_splited[1]
    else:
        return feature_column
def extract_scores_labels_indexes_from_files(paths):
    '''Given a list of paths for csv files containing models predicitions scores
extract the scores and combine them into one np array.
The last line the file should contain the indexes of each data point
The second raw from the end the actual labels '''
    all_scores = []
    for path in paths:
        # Read the file
        scores_lbl_idx = np.genfromtxt(path, delimiter=',')
        scores = scores_lbl_idx[:-2] # keep only the scores
        all_scores.append(scores) # Add the scores to the scores array
    # get indices and labels
    indexes = scores_lbl_idx[-1]
    y_test = scores_lbl_idx[-2]
    # Concate the arrays
    all_scores = np.concatenate(all_scores)
    return all_scores,y_test,indexes           

def get_k_choose_n(n,k):
    indices_combinations = list(combinations(range(1,n+1), k))
    return indices_combinations

def get_X_random_indices(N, k, X):
    '''Create X random combinations sized k out of N
    Example: N = 50, k = 3, X = 200
    Will create 200 random combinations of 3 indices out of 50 without replacement
    --------------
    Returns a list of X random combinations of k indices out of N'''
    # Create a list of indices from 0 to n_models
    indices = list(range(N))
    all_indices = []
    # Create X random combinations of k indices
    for j in range(X):
        random_indices = np.random.choice(indices, k, replace=False)
        all_indices.append(random_indices)
    return all_indices


## FILES
def remove_dir_recursivly(dir_path):
    try:
        rmtree(dir_path)
        print(f"Directory '{dir_path}' and its contents have been removed.")
    except Exception as e:
        print(f"Error: {e}") 

def create_paths(folder):
    '''Create paths list off all the files/folders in the given folder'''
    paths = []
    for path in os.listdir(folder):
        paths.append(os.path.join(folder,path))
    return paths

'''create folder in spesefic path'''
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("created new folder: ", path)

def keep_only_folders(paths_list):
    '''Given list of paths return only folders from the list'''
    return [path for path in paths_list if os.path.isdir(path)]

def validate_path(path):
    '''Validate if path exists or not'''
    return os.path.exists(path)

def find_target_folders(root_dir, target_subdirs):
    '''This function will iterate the root directory and return paths to the target folders
    '''
    target_folders = []
    for current_dir, dirs, files in os.walk(root_dir):
        # Check if both "Scores" and "Combi" are in the current directory
       if all(subdir in dirs for subdir in target_subdirs):
            target_folders.append(current_dir)
    return target_folders

def extract_ensmbel_combi_inner_paths(base_path):
    '''This function will iterate the base path:
    Base path -> partitions -> inner folders (number of ensmbels) - > Combi
    --------
    Returns a list of paths to the Combi folders from each inner folder'''
    path_lists = []
    for partition in os.listdir(base_path): # iterate partition
        partition_path = os.path.join(base_path,partition)
        for n_ensmbels_path in os.listdir(partition_path): # iterate inner folders
            parti_ensmbels_path = os.path.join(partition_path,n_ensmbels_path)
            if os.path.isdir(os.path.join(parti_ensmbels_path,"Combi")): # if Combi folder exists
                path_lists.append(parti_ensmbels_path)
    return path_lists


def get_bed_folder(bed_parent_folder):
    ''' function iterate on bed folder and returns a list of tuples:
    each tuple: [0] - folder name [1] - list of paths for the bed files in that folder.'''  
    # create a list of tuples - each tuple contain - folder name, folder path inside the parent bed file folder.
    subfolders_info = [(entry.name, entry.path) for entry in os.scandir(bed_parent_folder) if entry.is_dir()]
    # Create a new list of tuples with folder names and the information retrieved from the get bed files
    result_list = [(folder_name, get_bed_files(folder_path)) for folder_name, folder_path in subfolders_info]
    return result_list

def get_bed_files(bed_files_folder):
        
    '''function retrives bed files
    args- bed foler
    return list paths.'''
    bed_files = []
    for foldername, subfolders, filenames in os.walk(bed_files_folder):
        for name in filenames:
            # check file type the narrow,broad, bed type. $ for ending
            if re.match(r'.*(\.bed|\.narrowPeak|\.broadPeak)$', name):
                bed_path = os.path.join(foldername, name)
                bed_files.append(bed_path)
    return bed_files

def save_intersect_guides(dataset_1, dataset_2, target_col_1, target_col_2, output_name):
    '''Given two datasets and the target columns to compare
    Save the intersection of the target columns at output_name'''
    target_1 = set(dataset_1[target_col_1])
    target_2 = set(dataset_2[target_col_2])
    intersect = target_1.intersection(target_2)
    intersect = list(intersect)
    intersect_df = pd.DataFrame(intersect, columns=['Intersect_guides'])
    intersect_df.to_csv(output_name, index=False)
    print(f"Intersect guides saved at {output_name}")


## This function might not be used in the future
def fill_tprs_fpr_by_max_length(tprs, fprs):
    '''This function find the maximum length of the tprs and fprs and fill the arrays that are shorter with their last value.
    This function is for plotting the tprs,fprs togther on the same plot.
    Args:
    1. tprs - list of tprs arrays
    2. fprs - list of fprs arrays
    -----------
    Returns the filled tprs and fprs arrays'''
    # Find max length index
    max_len_index = np.argmax([len(fpr) for fpr in fprs])
    max_length = len(fprs[max_len_index])
    # Fill other indexes with their last value
    for i in range(len(tprs)):
        if i != max_len_index:
            last_tpr = tprs[i][-1]
            last_fpr = fprs[i][-1]
            tprs[i] += [last_tpr] * (max_length - len(tprs[i]))
            fprs[i] += [last_fpr] * (max_length - len(fprs[i]))
    return tprs, fprs

# Function to calculate averages for NchooseK
def calculate_averages_partitions(df, K):
    '''Given a data frame with columns: Positives, Negatives, Guides_amount
    The function will calculate the average positives, negatives and guides amount for each union of K rows
    Args:
        df - data frame with columns: Positives, Negatives, Guides_amount
        K - number of rows to union'''
    combinations_results = list(combinations(df.index, K))
    averages = {"Positives": [], "Negatives": [], "Guides_amount": []}

    for combo in combinations_results:
        selected_rows = df.loc[list(combo)]
        avg_pos = selected_rows['Positives'].sum()
        avg_neg = selected_rows['Negatives'].sum()
        avg_amount = selected_rows['Guides_amount'].sum()
        averages['Positives'].append(avg_pos)
        averages['Negatives'].append(avg_neg)
        averages['Guides_amount'].append(avg_amount)
    averages = {key: np.array(value) for key, value in averages.items()}
    averages = {key: value.mean() for key, value in averages.items()}
    return averages

def union_partitions_stats(data_path):
    '''Given a data path with the partitions stats get the num of rows and print the average stats for each union of k rows
    Args:
        data_path - path to the data'''
    data = pd.read_csv(data_path)
    rows = len(data)
    for k in range(1, rows + 1):
        averages = calculate_averages_partitions(data, k)
        print(f"Union of {k} rows: Positives: {averages['Positives']}, Negatives: {averages['Negatives']}, Guides amount: {averages['Guides_amount']}")

### Numeric Validations
def validate_non_negative_int(number):
    if not isinstance(number, int):
        raise Exception("Number must be an integer")
    if number < 1:
        raise Exception("Number must be positive")
    return number


def print_dict_values(columns_config):
    '''This function prints the json columns.'''
    # Access constants from the dictionary
    for key, value in columns_config.items():
        print(f'{key}: {value}')

def concat_change_seq_df():
    old_cs = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/merged_csgs_withEpigenetic_ALL_indexed.csv")
    new_cs = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/merged_new_csgs_nobulges_withEpigenetic.csv")
   
    merged = pd.concat([old_cs,new_cs])
    print(f'{len(merged)}, {len(old_cs)}, {len(new_cs)}, {len(merged) == len(old_cs) + len(new_cs)}')
    merged = merged.drop(columns=['Align.sgRNA','Align.#Bulges','genomic_coordinate'])
    
    print(merged["Read_count"].isnull().sum())
    merged['Read_count'] = merged['Read_count'].fillna(merged['GUIDEseq_reads'])
    print(merged["Read_count"].isnull().sum())
    merged['Read_count'] = merged['Read_count'].fillna(merged['CHANGEseq_reads'])
    print(merged["Read_count"].isnull().sum())

    merged = merged.drop(columns=['CHANGEseq_reads','GUIDEseq_reads'])
    
    mer_value_counts = merged["Label"].value_counts()
    old_count = old_cs["Label"].value_counts()
    new_count = new_cs["Label"].value_counts()
    print(f'old: {old_count}, new: {new_count}, merged: {mer_value_counts}, ')
    print(f'old + new pos: {old_count[1] + new_count[1] == mer_value_counts[1]}')
    print(f'old + new neg: {old_count[0] + new_count[0] == mer_value_counts[0]}')
    print(merged["offtarget_sequence"].isnull().sum())
    print(merged["target"].isnull().sum())
    merged= merged.drop(columns=["Index"])
    merged["Index"] = merged.index
    merged.to_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivovitro_nobulges_withEpigenetic_indexed.csv",index=False)
if __name__ == "__main__":
    data = get_partition_information("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/partition_guides_78/Changeseq-Partition_vivo_vitro.csv",7)
    print(data)
    

    #union_partitions_stats("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Hendel-Partition_1.csv")
    # #list_50 = [i for i in range(2,51)]
    # list_50 = [50]
    # dict_50_only_seq = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Only_seq/1_partition_50/Combi",list_50)
    # dict_50_atac = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Atacseq/Combi",list_50)
    # dict_50_h3k4me3 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/H3K4me3/Combi",list_50)
    # dict_50_both = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Both/Combi",list_50)
    # aurocs_results = [np.mean(dict_50_only_seq[50],axis=0)[0],np.mean(dict_50_atac[50],axis=0)[0],np.mean(dict_50_h3k4me3[50],axis=0)[0],np.mean(dict_50_both[50],axis=0)[0]]
    # aurocs_stds = [np.std(dict_50_only_seq[50],axis=0)[0],np.std(dict_50_atac[50],axis=0)[0],np.std(dict_50_h3k4me3[50],axis=0)[0],np.std(dict_50_both[50],axis=0)[0]]
    # plot_ensemble_performance_mean_std(aurocs_results,aurocs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auroc by ensmbels - seq,epi","Auroc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    # auprcs_results = [np.mean(dict_50_only_seq[50],axis=0)[1],np.mean(dict_50_atac[50],axis=0)[1],np.mean(dict_50_h3k4me3[50],axis=0)[1],np.mean(dict_50_both[50],axis=0)[1]]
    # auprcs_stds = [np.std(dict_50_only_seq[50],axis=0)[1],np.std(dict_50_atac[50],axis=0)[1],np.std(dict_50_h3k4me3[50],axis=0)[1],np.std(dict_50_both[50],axis=0)[1]]
    # plot_ensemble_performance_mean_std(auprcs_results,auprcs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auprc by ensmbels - seq,epi","Auprc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    # n_ranks_results = [np.mean(dict_50_only_seq[50],axis=0)[2],np.mean(dict_50_atac[50],axis=0)[2],np.mean(dict_50_h3k4me3[50],axis=0)[2],np.mean(dict_50_both[50],axis=0)[2]]
    # n_ranks_stds = [np.std(dict_50_only_seq[50],axis=0)[2],np.std(dict_50_atac[50],axis=0)[2],np.std(dict_50_h3k4me3[50],axis=0)[2],np.std(dict_50_both[50],axis=0)[2]]
    # plot_ensemble_performance_mean_std(n_ranks_results,n_ranks_stds,["Only_seq","Atacseq","H3K4me3","Both"],"N-rank by ensmbels - seq,epi","N-rank","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    # give_tests = ['GAAGCATGACGGACAAGTACNGG', 'GCCGTGGCAAACTGGTACTTNGG', 'GCTTCGGCAGGCTGACAGCCNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GGTTTCACCGAGACCTCAGTNGG', 'GATTTCCTCCTCGACCACCANGG', 'GGGGCCACTAGGGACAGGATNGG', 'GAGCCACATTAACCGGCCCTNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGTCCTGAAGTGGACATANGG', 'GGAAACTTGGCCACTCTATGNGG', 'GATTTCTATGACCTGTATGGNGG', 'GAAGATGATGGAGTAGATGGNGG', 'GCACGTGGCCCAGCCTGCTGNGG', 'GGTACCTATCGATTGTCAGGNGG', 'GCGTGACTTCCACATGAGCGNGG', 'GATGCTATTCAGGATGCAGTNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GGCCCAGCCTGCTGTGGTACNGG', 'GACATTAAAGATAGTCATCTNGG', 'GGGGGGTTCCAGGGCCTGTCNGG', 'GAAGGTGGCGTTGTCCCCTTNGG', 'GGGTATTATTGATGCTATTCNGG', 'GGCAGAAACCCTGGTGGTCGNGG', 'GCTGCAGAAACAGCAAGCCCNGG', 'GGATTTCCTCCTCGACCACCNGG', 'GAAGGCTGAGATCCTGGAGGNGG']
    # guides = ['GGCCGAGATGTCTCGCTCCGNGG', 'GCTGCAGAAACAGCAAGCCCNGG', 'GGACAGTAAGAAGGAAAAACNGG', 'GAAGGTGGCGTTGTCCCCTTNGG', 'GATAACTACACCGAGGAAATNGG', 'GGGATCAGGTGACCCATATTNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GAAGGCTGAGATCCTGGAGGNGG', 'GCACGTGGCCCAGCCTGCTGNGG', 'GACATTAAAGATAGTCATCTNGG', 'GGGGCAGCTCCGGCGCTCCTNGG', 'GGAGAAGGTGGGGGGGTTCCNGG', 'GACACCTTCTTCCCCAGCCCNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GGCAGAAACCCTGGTGGTCGNGG', 'GTCCCCTCCACCCCACAGTGNGG', 'GTATGGAAAATGAGAGCTGCNGG', 'GCCCTGCTCGTGGTGACCGANGG', 'GGTACCTATCGATTGTCAGGNGG', 'GGCGCCCTGGCCAGTCGTCTNGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GAGGTTCACTTGATTTCCACNGG', 'GTCCCTAGTGGCCCCACTGTNGG', 'GTCTCCCTGATCCATCCAGTNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GTTTGCGACTCTGACAGAGCNGG', 'GCTTCGGCAGGCTGACAGCCNGG', 'GCATTTTCTTCACGGAAACANGG', 'GATGCTATTCAGGATGCAGTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GGTTTCACCGAGACCTCAGTNGG', 'GATTTCTATGACCTGTATGGNGG', 'GGCCACGGAGCGAGACATCTNGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GGACTGAGGGCCATGGACACNGG', 'GCGTGACTTCCACATGAGCGNGG', 'GCTGGTACACGGCAGGGTCANGG', 'GAAGATGATGGAGTAGATGGNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCCGTGGCAAACTGGTACTTNGG', 'GCTCGGGGACACAGGATCCCNGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GGCCCAGCCTGCTGTGGTACNGG', 'GGATTTCCTCCTCGACCACCNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GAAGCATGACGGACAAGTACNGG', 'GGTGGATGATGGTGCCGTCGNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGCAGGGCTGGGGAGAAGGNGG', 'GTCACCAATCCTGTCCCTAGNGG', 'GGGGGGTTCCAGGGCCTGTCNGG', 'GGCCCCACTGTGGGGTGGAGNGG', 'GAGCCACATTAACCGGCCCTNGG', 'GGGAACCCAGCGAGTGAAGANGG', 'GGAAACTTGGCCACTCTATGNGG', 'GGGTATTATTGATGCTATTCNGG', 'GAGTAGCGCGAGCACAGCTANGG']
    # give_tests_s = set(give_tests)
    # guides_s = set(guides)
    # print(len(give_tests_s.intersection(guides_s)))
    # idx_list=[idx for idx, guide in enumerate(guides) if guide not in []]
    # print(len(idx_list))
   
    
    
    pass
    # list_dcits_only_seq = {i : None for i in range(10,81,10)} # 10,20,30,40,50,60,70,80
    # for partition in list_dcits_only_seq.keys():
    #     temp_path = f"/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/CNN/Ensemble/Only_sequence/1_partition/1_partition_{partition}/Combi"
    #     list_dcits_only_seq[partition] = extract_combinatorical_results(temp_path,[partition])
    # list_dcits_only_seq = get_mean_std_from_ensmbel_results(list_dcits_only_seq)
    # plot_all_ensmbels_std(list_dcits_only_seq,"Only_seq")
    
    
    
    