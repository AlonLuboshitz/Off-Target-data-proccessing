'''
Module to interpret data and models
'''
import pandas as pd
import numpy as np
import os
import shap
from file_utilities import create_folder,create_paths
from features_engineering import generate_features_and_labels, extract_features
from correlation_2 import hypergeometric_test, feature_correlation
from features_and_model_utilities import get_feature_name
from plotting import plot_correlation
from k_groups_utilities import extract_guides_from_partition
from train_and_test_utilities import split_by_guides
from models import argmax_layer
##################### data #####################


def binary_feature_enrichment_by_partition(partitions, features_columns,label_column,output_path,data_path =None,partition_info_path=None):
    '''
    features = [
    'Chromstate_H3K27me3_peaks_binary', 
    'Chromstate_H3K27ac_peaks_binary', 
    'Chromstate_H3K9ac_peaks_binary', 
    'Chromstate_H3K9me3_peaks_binary', 
    'Chromstate_H3K36me3_peaks_binary', 
    'Chromstate_ATAC-seq_peaks_binary', 
    'Chromstate_H3K4me3_peaks_binary', 
    'Chromstate_H3K4me1_peaks_binary'
]
    output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Feature_correlations/Change-seq/Vivo-vitro"
    data_path ="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/vivovitro_nobulges_withEpigenetic_indexed_read_count_with_model_scores.csv"
    partition_info = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/partition_guides_78/Changeseq-Partition_vivo_vitro.csv"
                              
    binary_feature_enrichment_by_partition([1,2,3,4,5,6,7],features,"Read_count",output_path,data_path,partition_info)'''
    
    if data_path is None:
        raise RuntimeError("No data path is given")
    if partition_info_path is None:
        raise RuntimeError("No parition data path is given")
    data = pd.read_csv(data_path)
    partition_info = pd.read_csv(partition_info_path)
    output_path = os.path.join(output_path,"Binary")
    create_folder(output_path)
    for partition in partitions:
        partition_guides = extract_guides_from_partition(partition_info,partition)
        partition_data = data[data["target"].isin(partition_guides)]
        temp_path = os.path.join(output_path,f"{partition}_partition.csv")
        binary_feature_enrichment(features_columns,label_column,temp_path,partition_data)
    temp_path = os.path.join(output_path,"All_partitions.csv")
    binary_feature_enrichment(features_columns,label_column,temp_path,data=data)

def binary_feature_enrichment( features_columns, label_column,output_path, data= None, data_path = None):
    '''This function gets a features columns list, label column and data and:
    Creates a data with the enrichment of each feature compared to the label.
    '''
    # Read data
    if data is None:
        if data_path is None:
            raise RuntimeError("No data path is given")
        else:
            data = pd.read_csv(data_path)
            output_path = os.path.join(output_path,f"all_data.csv")
    positives = len(data[data[label_column] > 0])
    negatives = len(data[data[label_column] == 0])
    index = ['positive_peaks','negative_peaks','positive_enrichment','negative_enrichment','p_val','geo_fold_pos','geo_fold_negative','positives','negatives']
    enrichment_data_set = pd.DataFrame()
    enrichment_data_set["Index"] = index
    enrichment_data_set.set_index("Index",inplace=True)
    for feature in features_columns:
        feature_in_positive,feature_in_nagative = get_feature_enrichment(data,feature,label_column)
        total_feature = feature_in_positive + feature_in_nagative
        positive_enrichment = feature_in_positive/positives
        negative_enrichment = feature_in_nagative/negatives
        p_val,geo_fold_positive,geo_fold_negative = hypergeometric_test(None,None,None,(positives,negatives,total_feature,feature_in_positive),True)
        feature_str = get_feature_name(feature)
        enrichment_data_set[feature_str] = [feature_in_positive,feature_in_nagative,positive_enrichment,negative_enrichment,p_val,geo_fold_positive,geo_fold_negative,positives,negatives]
    enrichment_data_set.to_csv(output_path)
def get_feature_enrichment(data,feature, label_column):
    feature_in_positive = len(data[(data[feature] > 0 ) & (data[label_column] > 0)])
    feature_in_nagative = len(data[(data[feature] > 0 ) & (data[label_column] == 0)])
    return (feature_in_positive,feature_in_nagative)

def plot_feature_correlation( output_path,  feature_columns, label_column,data_path=None,data=None):
    '''This function will plot the correlation between the features and the label.
    The function will plot the scatter plot for each feature and the label.
    Args:
    1. data_path - path to the data.
    2. output_path - path to save the plots.
    3. feature_columns - columns with the features.
    4. label_column - column with the label.
    ------------
    Returns: None
    
    plot_feature_correlation("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism_with_model_scores.csv",
                             "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel/Feature_correlation/MOFF",["MOFF","GMT"],"Label")'''
    if label_column == "Label":
        log = False
    else: log = True
    # Calculate the correlation
    if data is None:
        if data_path is None:
            raise RuntimeError("No data path is given")
        else:
            data = pd.read_csv(data_path)
    correlation_dict = feature_correlation(data, feature_columns, label_column, log_label = log)
    # Plot the correlation
    for feature_nd_label, values in correlation_dict.items():
        r, p, x_values, y_values = values
        feature,label = feature_nd_label
        feature = feature.replace("_"," ")
        label = label.replace("_"," ")
        y_label = label.replace("Positive","") # remove positive from the label
        plot_correlation(x=x_values, y=y_values, x_axis_label=feature + " score", y_axis_label=y_label, r_coeff=r, p_value=p, title=feature + " " + label, output_path=output_path)


##################### model #####################

def get_shaply_values(model, x_background, explainer_type, num_of_points=None, specific_indices=None,only_seq=True):
    """
    Computes SHAP values for the given model using the specified explainer type.
    
    Args:
        model: Trained model to explain.
        x_background (numpy.ndarray or pandas.DataFrame): Background dataset for SHAP.
        explainer_type (str): Type of SHAP explainer to use ('deep', 'gradient', 'kernel').
        num_of_points (int, optional): Number of first indices to use from x_background.
        specific_indices (list, optional): Specific indices to extract from x_background.
    
    Returns:
        shap_values (list of numpy arrays): Computed SHAP values for each output class (or regression target).
    """
    
    additional_features = 0
    if not only_seq:
        x_background = extract_features(x_background, encoded_length= 600)
        additional_features = len(x_background[1])
    if explainer_type == 'deep':
        explainer = shap.Explainer(model, x_background)

        #explainer = shap.DeepExplainer(model, x_selected)
    elif explainer_type == 'gradient':
        explainer = shap.GradientExplainer(model, x_background)
    elif explainer_type == 'kernel':
        explainer = shap.KernelExplainer(model.predict, x_background)
    else:
        print('using default explainer: shap.Explainer')
        explainer = shap.Explainer(model, x_background)
    shap_values = explainer.shap_values(x_background)
    shap_values, feature_names = convert_features_names(shap_values, np.sum, 24, 25, additional_features)
    return shap_values,feature_names, explainer
def convert_features_names(shap_values, agg_function, seqeunce_length, bits_per_base, additional_features_length):
    '''
    Convert the shap values of all features into group of features
    by aggregating the values of each group of features
    Args:
        shap_values (list of numpy arrays): List of SHAP value arrays.
        agg_function (function): Aggregation function to use.
    Returns:
        list of numpy.ndarray: Aggregated SHAP values.
    '''
    grouping = [list(range(i * bits_per_base, (i + 1) * bits_per_base)) for i in range(seqeunce_length)]
    total_sequence_length = seqeunce_length * bits_per_base
    grouping.append(list(range(total_sequence_length , total_sequence_length + additional_features_length)))  
    feature_names = [f"Base_{i+1}" for i in range(24)] + ["epigenetics"]
    shap_values_grouped = np.array([shap_values[:, group].sum(axis=1) for group in grouping]).T
    return shap_values_grouped, feature_names


def average_shap_values(shap_values_list):
    """
    Computes the average SHAP values across multiple runs/models.
    
    Args:
        shap_values_list (list of numpy arrays): List of SHAP value arrays from different models/runs.
    
    Returns:
        numpy.ndarray: Averaged SHAP values with the same shape as input SHAP values.
    """
    shap_values_array = np.array(shap_values_list)
    return np.mean(shap_values_array, axis=0)

def median_shap_values(shap_values_list):
    """
    Computes the median SHAP values across multiple runs/models.
    
    Args:
        shap_values_list (list of numpy arrays): List of SHAP value arrays from different models/runs.
    
    Returns:
        numpy.ndarray: Median SHAP values with the same shape as input SHAP values.
    """
    shap_values_array = np.array(shap_values_list)
    return np.median(shap_values_array, axis=0)

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
    models = []
    models_path = create_paths(model_path)
    for model_path in models_path:
        if model_type == "deep":
            model = tf.keras.models.load_model(model_path,custom_objects={'argmax_layer': argmax_layer})
            models.append(model)

        else:
            pass
    return models

def get_data(data_path):
    '''
    Loads the data from the given path
    Uses generate_features_and_labels function to get x,y,guides data
    
    Args:
        data_path (str): path to the data
    Returns:
        x,y,guides
        x: list of arrays- each array is all (gRNA,OTS) pairs.
        y: list of arrays - each array is the labels for the pairs.
        guides: list of guides
    '''
    Columns_dict = {
    "TARGET_COLUMN": "target",
    "REALIGNED_COLUMN": "realigned_target",
    "OFFTARGET_COLUMN": "offtarget_sequence",
    "CHROM_COLUMN": "chrom",
    "START_COLUMN": "chromStart",
    "END_COLUMN": "chromEnd",
    "BINARY_LABEL_COLUMN": "Label",
    "REGRESSION_LABEL_COLUMN": "Read_count"
}
    Columns_dict['Y_LABEL_COLUMN'] = Columns_dict['BINARY_LABEL_COLUMN']
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    x,y,guides = generate_features_and_labels(data_path=data_path,manager=None,
                                              if_bp=False,if_only_seq=True,if_seperate_epi=False,
                                              epigenetic_window_size=0,features_columns=None,
                                              if_data_reproducibility=False,columns_dict=Columns_dict,
                                              sequence_coding_type=2,if_bulges=True)
    return x,y,guides
def run_shap(model_path,data_path,explainer_type,output_path,
             num_of_points=None,specific_indices=None, specific_guides=None):
    '''
    Runs on the data and model given and extract shap values
    Plots the bars, beeswarm and waterfall plots
    
    Args:
        model_path (str): path to the model/folder of models
        data_path (str): path to the data
        explainer_type (str): type of the explainer to use
        output_path (str): path to save the plots
        num_of_points (int, optional): Number of first indices to use from x_background.
        specific_indices (list, optional): Specific indices to extract from x_background.
        specific_guides (list, optional): Specific guides to extract from x_background.
    '''
    models = get_model(model_path,"deep")
    x_background,y,guides = get_data(data_path)
    if specific_guides is not None:
        x_background,y,guides = split_by_guides(guides,specific_guides,x_background,y)
    elif specific_indices is not None:
        x_selected = x_background[specific_indices]
    else:
        x_selected = x_background
    if num_of_points is not None:
        x_selected = x_background[:num_of_points]
    shap_values_list = []
    for model in models:
        shap_values = get_shaply_values(model, x_selected, explainer_type, num_of_points, specific_indices)
        shap_values_list.append(shap_values)
    for val,feature_names,explainer in shap_values_list:
        shap_values_expl = shap.Explanation(values=val[0], 
                                    base_values=explainer.expected_value, 
                                    data=x_selected,feature_names=feature_names)
        shap.plots.waterfall(shap_values_expl)
        shap_values_expl = shap.Explanation(values=val, 
                                    base_values=explainer.expected_value, 
                                    data=x_selected,feature_names=feature_names)
        shap.bar_plot(shap_values_expl)
        shap.plots.beeswarn(shap_values_expl)
    
    
if __name__ == "__main__":
    model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1/model_1.keras"
    data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    explainer_type = "deep"
    specific_guides = ["GGACTGAGGGCCATGGACACNGG"]
    number_of_points = 2
    run_shap(model_path=model_path,data_path=data_path,explainer_type=explainer_type,
             output_path=None,num_of_points=number_of_points,specific_guides=specific_guides)
