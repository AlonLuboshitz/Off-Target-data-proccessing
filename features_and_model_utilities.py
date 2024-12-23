'''
This module is used to define parameters needed for features and model training.
'''
############# !!!!!!! NEED TO ADD EPIGENETIC BY BASE PAIR and STUFF LIKE THAT !!!!!!! #############
def get_full_enconding_parameters_(if_bulges = None):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the full enconding.
    '''
    if if_bulges is None:
        raise ValueError("if_bulges  must be provided.")
    if if_bulges:
        seq_len = 24
        nuc_num = 5 #ATCG-
    else:
        seq_len = 23
        nuc_num = 4 #ATCG
    return (seq_len, nuc_num) 

def get_PiCRISPR_enconding_parameters(if_bulges = None):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the PiCRISPR enconding.
    '''
    if if_bulges is None:
        raise ValueError("if_bulges  must be provided.")
    if if_bulges:
        raise ValueError("PiCRISPR encoding does not support bulges")
    else:
        seq_len = 23
        nuc_num = 6 #ATCG
    return (seq_len, nuc_num)

def get_encoding_parameters(coding_type, if_bulges):
    '''
    Gets the length of the sequence and the number of nucleotides in the sequence for the given encoding type.
    '''
    if coding_type == 1:
        return get_PiCRISPR_enconding_parameters(if_bulges)
    elif coding_type == 2:
        return get_full_enconding_parameters_(if_bulges)
    else:
        raise ValueError("Invalid coding type")
    
#################
'''
Epigenetic feature utilities
'''
           

def parse_feature_column_dict(feature_column_dict):
    '''
    This function adds all the features to a list called All_features.
    If there is only one list of features it just return this dict.
    If there are subsets of features from the given group it skips as they inserted to the dict already.
    '''
    all_features = []
    if len(feature_column_dict) == 1:
        return feature_column_dict
    for keys,features in feature_column_dict.items():
        if "Subset" in keys:
            continue
        all_features.extend(features)
    feature_column_dict["All_features"] = all_features
    return feature_column_dict


def get_features_columns_args_ensembles(runner = None,file_manager = None, t_guides = None, 
                              model_base_path = None, ml_results_base_path = None,
                                n_models = None, n_ensmbels = None, features_dict = None, multi_process= False):
    '''
    This creates a list of arguments for training/testing the models with features.
    ARGS:
    runner: Runner object
    file_manager: FileManager object
    t_guides: list of guides for training/testing
    model_base_path: Path to save/get the models
    ml_results_base_path: Path to save the results
    n_models: Number of models to train
    n_ensmbels: Number of ensembles to train
    features_dict: Dictionary of features - {group: [features]}
    multi_process: If to run the models in parallel
    '''
    arg_list = []
    for group, features in features_dict.items():
        if len(features) > 1: # More then 1 feature in the group run all togther.
            arg_list.append((group, features,runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels, multi_process))
        if "Subset" in group or "All" in group:
            continue
        for feature in features:
            arg_list.append((group, [feature],runner, file_manager,t_guides,model_base_path,ml_results_base_path, n_models, n_ensmbels,multi_process))
    return arg_list    



############################## DEEP LEARNING UTILITIES ##############################
