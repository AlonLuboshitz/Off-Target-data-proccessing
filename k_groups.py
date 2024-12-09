import pandas as pd
import numpy as np
import os


def create_k_balanced_groups(dataset, target_column, label_column, k, output_name, seperate_grna_path, y_labels_tup = None, constrained_guides = None):
    '''Given y_labels and k, create k groups with rougly eqaul amount of labels in each group
    Args: 
        dataset - path for the dataset
        target_column - name of the target column
        k - number of groups
        output_name - name of the complete output file
        seperate_grna_path - path to save each group seperatly
        y_labels_tup - list of labels for each guide (optional)
        if y_labels_tup given it should be given the a list of guides!
        constrained_guides - tuple of:
        (list of guides need to be kept in one group (Usauly for test set), bool- true to keep them sololy, false to keep them with more guides)
         NOTE: By defualt they will be kept sololy
        -----------
        Saves the groups in a csv file containing:
        Positives, Negatives, list of guides.
        -----------
        Save each gRNA group separtley in txt file
        -----------
        Example:     
        cons = (['GCTGGTACACGGCAGGGTCANGG', 'GGCGCCCTGGCCAGTCGTCTNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GCAGCATAGTGAGCCCAGAANGG'], True)
        create_k_balanced_groups("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivovitro_nobulges_withEpigenetic_indexed_read_count_with_model_scores.csv","target","Label",7
                             ,"Changeseq-Partition","/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/partition_guides_78",constrained_guides=cons)

        '''
    if y_labels_tup is None: # y_labels not given, load them from the dataset
        dataset = pd.read_csv(dataset)
        guides = set(dataset[target_column]) # get unuiqe guides
        y_labels = [dataset[dataset[target_column] == guide][label_column].values for guide in guides] # get the labels for each guide
    else : # y_labels given
        guides = y_labels_tup[1]
        y_labels = y_labels_tup[0]
    # Get the sum of the labels for each guide and sort it by descending order        
    sum_labels = [np.sum(array > 0) for array in y_labels] 
    print(f'Total positives: {sum(sum_labels)}')     
    sorted_indices = np.argsort(sum_labels)[::-1]
    guides = list(guides)
    keep_constrained_sololy = False
    if constrained_guides is not None:
        if constrained_guides[1]:
            keep_constrained_sololy = True
        constrained_guides = [guides.index(guide) for guide in constrained_guides[0]]
        sorted_indices = [i for i in sorted_indices if i not in constrained_guides] # remove the constrained guides from the sorted indices
        
    # init K groups and fill them with features by the sorted indices
    k_groups = fill_k_groups_indices(k, sum_labels = sum_labels, sorted_indices = sorted_indices, constrained_grna_indices=constrained_guides,keep_constraied_sololy=keep_constrained_sololy)
    write_guides_seperatley(k_groups, guides, seperate_grna_path)
    save_complete_partition_information(k_groups, y_labels, guides, output_name)

def fill_k_groups_indices(k, sum_labels, sorted_indices, constrained_grna_indices = None, keep_constraied_sololy = False):
        '''Greedy approch to fill k groups with ~ equal amount of labels
        Getting sum of labels for each indice and sorted indices by sum, filling the groups
        from the biggest amount to smallest adding to the minimum group
        Args:
            k - number of groups
            sum_labels - sum of labels for each indice
            sorted_indices - indices sorted by sum of labels
            constrained_grna_indices - list of indices that need to be in the same group
            -----------
            Returns a list of lists with the indices of the guides in each group'''
        
        if k > len(sorted_indices):
            raise RuntimeError("K value is bigger than the amount of guides")
        # Create k groups with 1 indice each from the sorted indices in descending order
        if constrained_grna_indices is not None:
            groups = [[sorted_indices[i]] for i in range(k-1)] # create k-1 groups
            if not keep_constraied_sololy: #  append the list of the constrained indices for k groups. 
                groups.append(constrained_grna_indices) 
            k -= 1 # update k value
        else : groups = [[sorted_indices[i]] for i in range(k)] 
        
        for index in sorted_indices[k:]: # Notice [K:] to itreate over the remaining indices
        # Find the group with the smallest current sum
            min_sum_group = min(groups, key=lambda group: sum(sum_labels[i] for i in group), default=groups[0])
            # Add the series to the group with the smallest current sum
            min_sum_group.append(index)
        if keep_constraied_sololy:
            groups.append(constrained_grna_indices)
        return groups
def write_guides_seperatley(k_groups, guides, output_path):
    '''Write each guides group to a txt file in the output path with train_guides_{i}_partition.txt
    Write each guides group diffrence from all guides to a txt file in the output path with tested_guides_{i}_partition.txt
    Args:
        k_groups - list of lists with the indices of the guides
        guides - list of all the guides
        output_path - path to save the groups'''
    
    for i, group in enumerate(k_groups): # iterate on each group
        train_temp_path = os.path.join(output_path,f"Train_guides_{i+1}_partition.txt")
        test_temp_path = os.path.join(output_path,f"Test_guides_{i+1}_partition.txt")
        all_indexes = set(range(len(guides)))
        with open(test_temp_path, "w") as test_file:
            for index in group:
                if index == group[-1]:
                    test_file.write(guides[index])
                else:
                    test_file.write(guides[index] + ", ")
        test_group_indexes = list(all_indexes - set(group))
        with open(train_temp_path, "w") as train_file:
            for index in test_group_indexes:
                if index == test_group_indexes[-1]:
                    train_file.write(guides[index])
                else:
                    train_file.write(guides[index] + ", ")
                    
        

def save_complete_partition_information(k_groups, labels, guides, output_name):
    '''Save the complete partition information to a csv file
    Args:
        k_groups - list of lists with the indices of the guides in each group
        labels - list of all the labels
        guides - list of all the guides
        output_name - name of the output file'''
    # Create a list of dictionaries with the information
    complete_info = []
    
    for i, group in enumerate(k_groups):
        group_info = {}
        group_info["Partition"] = i + 1
        group_info["Positives"] = sum(np.sum(labels[index] > 0) for index in group)
        group_info["Negatives"] = sum(np.sum(labels[index] <= 0) for index in group)
        group_info["Guides_amount"] = len(group)
        group_info["Guides"] = [guides[index] for index in group]
        complete_info.append(group_info)
    # Save the information to a csv file
    complete_info_df = pd.DataFrame(complete_info)
    complete_info_df.to_csv(f'{output_name}.csv', index=False)
    print(f"Complete partition information saved at {output_name}")

def get_partition_information(partition_summary_path, partition_number):
    '''This function returns for the partition number the positives, negatives and guides amount
    Args:
    1. partition_summary_path - path to the partition summary file
    2. partition_number - number of the partition
    -----------
    Returns: Positives, Negatives, Guides amount'''
    partitions_data = pd.read_csv(partition_summary_path)
    partition_data = partitions_data[partitions_data["Partition"] == partition_number]
    return {
        'Positives': partition_data['Positives'].values[0],
        'Negatives': partition_data['Negatives'].values[0],
        'sgRNAs': partition_data['Guides_amount'].values[0]
    }


def get_k_groups_ensemble_args(partitions, models, ensembles, multi_process = False, other_feature_columns = None, method = None):
    if not method: # None
        raise ValueError("Method is not given")
    elif method == 1: # only sequence features
        return get_k_groups_ensemble_args_only_seq(partitions, models, ensembles, multi_process)
    elif method == 2: # epigenetic features
        return get_k_groups_ensemble_args_epi_features(partitions, models, ensembles, multi_process)
    elif method == 5: # other features
        return get_k_groups_ensebmle_args_other_features(partitions, models, ensembles, multi_process, other_feature_columns)
    else:
        raise ValueError("Method is not valid")
def get_k_groups_ensemble_args_only_seq(partitions, models, ensembles,multi_process=False):
    '''This function will create arguments for the k_groups with ensemble only sequence features.'''
    args = []
    for partition in partitions:
        cross_val_params = (models, ensembles, [partition])
        model_params = (None,None,3,1) # 3 - ensebmle, 1- only-seq
        args.append((model_params,cross_val_params,multi_process ))
    return args

def get_k_groups_ensemble_args_epi_features(partitions, n_models, n_ensmbels, multi_process=False):
    '''This function creates the arguments for the k_groups with ensemble for epigenetic features.
    The function will return a list of arguments for each partition in the partition list.'''
    multi_process_args = []
    for partition in partitions:
        cross_val_params = (n_models, n_ensmbels, [partition])
        model_params = (None,None,3,2) # 3 - ensemble, 2 - epigenetic features
        multi_process_args.append((model_params,cross_val_params,multi_process)) # model_params, cross_val_params, multi_process
    return multi_process_args

def get_k_groups_ensebmle_args_other_features(partitions, n_models, n_ensmbels, multi_process= False, other_feature_columns = None):
    if other_feature_columns is None:
        raise ValueError("Other feature columns are not given")
    multi_process_args = []
    for partition in partitions:
        cross_val_params = (n_models, n_ensmbels, [partition])
        multi_process_args.append((None,cross_val_params,False, other_feature_columns))
    return multi_process_args
