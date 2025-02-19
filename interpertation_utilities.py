'''
This module contain helper function and utilities for model and data interpertability.
'''
import numpy as np
from features_engineering import extract_features, generate_features_and_labels
from file_utilities import create_paths


######## DATA ########
def nucleotides_for_heatmap():
    '''
    Create the row labels (nucleotides) per position for the heatmap.
    Returns:
        row_labels (list): List of row labels.
        x_ticks (list): List of x-ticks.'''
    import itertools
    nucleotides_product = list(itertools.product(*(["ACGT-"] * 2)))
    row_labels = [f"{a}:{b}" for a, b in nucleotides_product]
    x_ticks = list(range(1, 25))
    return row_labels, x_ticks
def get_model(model_path, model_type):
    '''
    Loads the model from the given path
    
    Args:
        model_path (str): path to model/folder of models
        model_type (str): type of the model - deep,ml
    Returns:
        models (list): list of models    
    '''
    import tensorflow as tf
    from models import argmax_layer
    models = []
    models_path = create_paths(model_path)
    for model_path in models_path:
        if model_type == "deep":
            model = tf.keras.models.load_model(model_path,custom_objects={'argmax_layer': argmax_layer})
            models.append(model)

        else:
            pass
    return models

def get_data(data_path, only_seq):
    '''
    Loads the data from the given path
    Uses generate_features_and_labels function to get x,y,guides, off-targets.
    
    Args:
        data_path (str): path to the data
        only_seq (bool): If True, only the sequence features will be used otherwise split to sequence and epigenetics.
    Returns:
        x,y,guides, off targets
        x: list of arrays- each array is all (gRNA,OTS) pairs.
        y: list of arrays - each array is the labels for the pairs.
        guides: list of guides
        otss: list of off-targets
    '''
    Columns_dict = {
    "TARGET_COLUMN": "target",
    "REALIGNED_COLUMN": "realigned_target",
    "OFFTARGET_COLUMN": "offtarget_sequence",
    "CHROM_COLUMN": "chrom",
    "START_COLUMN": "chromStart",
    "END_COLUMN": "chromEnd",
    "BINARY_LABEL_COLUMN": "Label",
    "REGRESSION_LABEL_COLUMN": "Read_count",
    "MISMATCH_COLUMN": "missmatches",
    "BULGES_COLUMN": "bulges"
}
    Columns_dict['Y_LABEL_COLUMN'] = Columns_dict['BINARY_LABEL_COLUMN']
    features = None
    if not only_seq:
        features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    x,y,guides,otss = generate_features_and_labels(data_path=data_path,manager=None,
                                              if_bp=False,if_only_seq=only_seq,if_seperate_epi=False,
                                              epigenetic_window_size=0,features_columns=features,
                                              if_data_reproducibility=False,columns_dict=Columns_dict,
                                              sequence_coding_type=2,if_bulges=True,return_otss=True, exclude_ontarget=True)
    return x,y,guides,otss

def filter_data_for_interpertation(x_background, y,  sgrna_otss, only_seq = False,
                        specific_indices = None, number_of_points = 200):
    '''
    Splits the data by guide in the guides list.
    each split will be interperated.
    Args:
        x_background (list of arrays): list of all ENCODED gRNA-OT pairs, each list is for a different sgRNA.
        y (list of arrays): list of all labels
        otss_dict (np.array 2D): 1d- all samples, 2d - all (gRNA,OT) pairs. 
        spesific_indices (list, optional): Specific indices to extract from x_background. If given these indices will be returned.
    Returns:'''
    
    if specific_indices is not None:
        pass #NOTE: ADD SPESIFIC INDICES WITH FEATURE ENGINGERRING FUNCTION
    
    additional_features = 0
    if not only_seq:
        x_background = extract_features(x_background, encoded_length= 600)
        additional_features = x_background[1].shape[1]
    
    # NOTE: SAMPLE OUT NEGATIVES (NOT BY STARTIFYING - NEED TO COMPLETE)
    sampled_indices = get_sampled_indices(y, number_of_points = number_of_points)
    if isinstance(x_background,list):
        x_selected = [x[sampled_indices] for x in x_background]
    else:
        x_selected = x_background[sampled_indices]
    sgrna_otss = sgrna_otss[sampled_indices]
    return x_selected, sgrna_otss, additional_features
    
    
def get_sampled_indices(y, number_of_points = None):
    '''
    Sample all positives and randomly sample negatives to complete the gap to number of points
    if number of points is None than balanced amount of positive and negatives will be returned.
    Args:
        y (np.array): labels
        number_of_points (int, optional): Total number of samples.
            None: Balanced amount of positive and negatives will be returned.
            0: all positives will be returned.
            >0: number of points to sample.
    Returns:
        list of indices
    '''
    positive_indexes = np.where(y == 1)[0]
    negative_indexes = np.where(y == 0)[0]
    positive_number = len(positive_indexes)
    if number_of_points is None: # balanced
        print("no number of points given, return balanced positives and negatives")
        negative_indexes = np.random.choice(negative_indexes, positive_number, replace=False)
        return np.concatenate((positive_indexes, negative_indexes))
    elif number_of_points <= positive_number:
        print("total number of points is smaller than the number of positives return all positives")
        return positive_indexes
    negative_number = number_of_points - positive_number # else sample negatives    
    random_negative_indices = np.random.choice(negative_indexes, negative_number, replace=False)
    return np.concatenate((positive_indexes, random_negative_indices))
    

######## SHAP ########
  
def convert_features_names(shap_values, agg_function, seqeunce_length, bits_per_base, additional_features_length = 0):
    '''
    Convert the shap values of all features into group of features
    by aggregating the values of each group of features
    Args:
        shap_values (list of numpy arrays): List of SHAP value arrays.
        agg_function (function): Aggregation function to use.
    Returns:
        list of numpy.ndarray: Aggregated SHAP values.
    '''
    original_values = shap_values.values  # Shape: (num_samples, N)
    groups = [list(range(i * bits_per_base, (i + 1) * bits_per_base)) for i in range(seqeunce_length)]
    feature_names = [f"Base_{i+1}" for i in range(seqeunce_length)]
    total_sequence_length = seqeunce_length * bits_per_base
    if additional_features_length > 0:
        groups.append(list(range(total_sequence_length , total_sequence_length + additional_features_length)))  
        feature_names.append("epigenetics")
    # Aggregate SHAP values by summing grouped features
    grouped_shap_values = np.zeros((original_values.shape[0], len(groups)))
    for i, indices in enumerate(groups):
        grouped_shap_values[:, i] = np.sum(original_values[:, indices], axis=1)
    # Update feature names
    shap_values.values = grouped_shap_values
    shap_values.feature_names = feature_names
    shap_values.data = shap_values.data[:, [g[0] for g in groups]]
    
    
    return shap_values