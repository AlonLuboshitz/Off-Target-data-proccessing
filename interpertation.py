'''
Module to interpret data and models
'''
import pandas as pd
import numpy as np
import os
import shap
import Levenshtein 
from file_utilities import create_folder
from Data_labeling_and_processing import remove_unwanted_samples
from correlation_2 import hypergeometric_test, feature_correlation
from features_and_model_utilities import get_feature_name
from plotting import plot_correlation, plot_subplots
from k_groups_utilities import extract_guides_from_partition
from train_and_test_utilities import keep_intersect_guides_indices
from interpertation_utilities import *

import tensorflow as tf

COLUMNS = {
    "TARGET_COLUMN": "target",
    "REALIGNED_COLUMN": "realigned_target",
    "OFFTARGET_COLUMN": "offtarget_sequence",
    "CHROM_COLUMN": "chrom",
    "START_COLUMN": "chromStart",
    "END_COLUMN": "chromEnd",
    "BINARY_LABEL_COLUMN": "Label",
    "REGRESSION_LABEL_COLUMN": "Read_count"
}

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

def get_number_of_mismatches_per_position(data_frame, target_column, offtarget_column, 
                                          bulges_column = "bulges", mismatches_column = "missmatches"):
    #NOTE: FIX MISMATCH COUNT
    '''
    Extract the total number of mismatches in each position between the sgRNA and the off-target sequences
    Args:
        data_frame (pd.DataFrame): pandas df object
        target_column (str): column name of the sgRNA
        offtarget_column (str): column name of the off-target sequence
    Returns:
        dictionary: key - position, value - number of mismatches in this position
        '''
    mismatch_only_data = remove_unwanted_samples(data_frame, bulges_column,0,True , "==")
    bulges_data = remove_unwanted_samples(data_frame,bulges_column,0 ,True,">")
    if (len(bulges_data) + len(mismatch_only_data)) != len(data_frame):
        raise ValueError("Data is not correctly seperated")
    mismatch_counts = np.zeros(23, dtype=int)  # positions 1-23 will be indexed from 0-22
    rna_bulges = np.zeros(24, dtype=int)
    dna_bulges = np.zeros(24, dtype=int)
    for sgrna,off_target in zip(mismatch_only_data[target_column],mismatch_only_data[offtarget_column]):
        mismatch_counts = update_mismatch_counts(sgrna,off_target,mismatch_counts)
    df = pd.DataFrame(columns=['sg','ot'])
    for index,(sgrna,off_target) in enumerate(zip(bulges_data[target_column],bulges_data[offtarget_column])):
        sgrna,off_target,rna_bulges,dna_bulges = update_bulges_and_remove(sgrna,off_target,rna_bulges,dna_bulges)
        df.loc[index,['sg','ot']] = [sgrna,off_target]
        mismatch_counts = update_mismatch_counts(sgrna,off_target,mismatch_counts)
    print((df['sg'].str[-3] == "N").sum())

    mismatch_counts[20] = 0  # Ignore the PAM position  
def update_bulges_and_remove(sgrna, offtarget, rna_bulges_array, dna_bulges_array):
    """
    Update the insertions or deletions in the gRNA/off-target sequences and remove them from the sequences
    """
    if len(sgrna) != len(offtarget):
        raise ValueError("length of sgrna and offtarget are not equal")
    updated_sg_rna = ""
    updated_ot = ""
    temp_index = 0
    for index,sg_char,ot_char in zip(range(len(sgrna)),sgrna,offtarget):
        if sg_char != "-" and ot_char != "-":
            continue
        else:
            updated_sg_rna += sgrna[temp_index:index]
            updated_ot += offtarget[temp_index:index]
            temp_index = index + 1

                
            if sg_char == "-" and ot_char =="-":
                rna_bulges_array[index] += 1
                dna_bulges_array[index] += 1
            elif sg_char == "-": # insertion
                rna_bulges_array[index] += 1
                

            elif ot_char == "-": # deletion
                dna_bulges_array[index] += 1
                updated_ot += sgrna[index:index+1]
                updated_sg_rna += sgrna[index:index+1]
    updated_sg_rna += sgrna[temp_index:index+1]
    updated_ot += offtarget[temp_index:index+1]          

    return updated_sg_rna, updated_ot, rna_bulges_array, dna_bulges_array
    
def update_mismatch_counts(sgrna, offtarget, mismatch_counts):
    """
    Update the mismatch counts for each position in the sgRNA and off-target sequences
    
    Args:
        data_frame (pd.DataFrame): Dataframe containing the sgRNA and off-target sequences.
        mismatch_counts (np.array): np.array containing the mismatch counts for each position.
        target_column (str): Column name of the sgRNA.
        offtarget_column (str): Column name of the off-target sequence.""" 
    
    if len(sgrna) != 23 or len(offtarget) != 23:
        raise ValueError("length of sgrna or offtarget is not 23")
    edits = Levenshtein.opcodes(sgrna, offtarget)
    for tag, start1, end1, start2, end2 in edits:
        if tag == "replace":
            mismatch_counts[start1:end1] += 1  # Increase counts for replaced positions
    return mismatch_counts

def main_data():
    data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    data_frame = pd.read_csv(data_path)
    data_frame = data_frame[data_frame['target'].isin(specific_guides)]
    get_number_of_mismatches_per_position(data_frame, COLUMNS["REALIGNED_COLUMN"], COLUMNS["OFFTARGET_COLUMN"])
##################### MODEL INTERPERTABILITY #####################

##################### SHAP #####################

def get_shaply_values(model, x_background, explainer_type, x_selected = None):
    """
    Computes SHAP values for the given model using the specified explainer type.
    
    Args:
        model: Trained model to explain.
        x_background (numpy.ndarray or pandas.DataFrame): Background dataset for SHAP.
        explainer_type (str): Type of SHAP explainer to use ('deep', 'gradient', 'kernel').
        x_selected (numpy.ndarray or pandas.DataFrame, optional): Selected dataset to explain.
    
    Returns:
        shap_values (list of numpy arrays): Computed SHAP values for each output class (or regression target).
    """
    
    
    if explainer_type == 'deep':
        # if isinstance(x_background,list):
        #     x_background = [x[:100] for x in x_background if x.shape[0] > 100]
        # elif x_background.shape[0] > 100:
        #     x_background = x_background[:100]
        #explainer = shap.Explainer(model, x_background)
        explainer = shap.explainers.Permutation(model,x_background ,max_evals = 15000)
        #explainer = shap.DeepExplainer(model, x_background)
    elif explainer_type == 'gradient':
        explainer = shap.GradientExplainer(model, x_background)
    elif explainer_type == 'kernel':
        explainer = shap.KernelExplainer(model.predict, x_background)
    else:
        print('using default explainer: shap.Explainer')
        explainer = shap.Explainer(model, x_background)
    if x_selected is None:
        x_selected = x_background
    shap_values = explainer(x_selected)
    return shap_values

def transform_to_heatmap(shap_values, seqeunce_length, bits_per_base, additional_features_length = 0):
    """
    Transforms SHAP values into a 2D matrix for heatmap representation.
    """
    if isinstance(shap_values,shap.Explanation):
        shap_values = shap_values.values
    min_shap = shap_values.min()
    max_shap = shap_values.max()
    epigenetics_values = None
    if additional_features_length > 0: # split the shap values to sequence and epigenetics
        sequence_values = shap_values[:,seqeunce_length * bits_per_base]
        epigenetics_values = shap_values[:,seqeunce_length * bits_per_base:]
    else: sequence_values = shap_values
    if sequence_values.ndim == 1:
        sequence_values = sequence_values.reshape(1,seqeunce_length , bits_per_base)
    elif sequence_values.ndim == 2:
        sequence_values = sequence_values.reshape(sequence_values.shape[0],seqeunce_length , bits_per_base)
    else:
        raise ValueError("SHAP values should be 1D or 2D")
    return sequence_values, epigenetics_values, min_shap, max_shap

def run_shap(model_path,data_path,explainer_type,output_path,
             num_of_points=None,specific_indices=None, specific_guides=None,
             only_seq=False, plot_all_guides = True, plot_single_guides = True):
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
        only_seq (bool, optional) defualt True: If True, only the sequence features will be used otherwise split to sequence and epigenetics.
    '''
    if not (plot_all_guides or plot_single_guides):
        raise ValueError("At least one of the plot options should be True")
    models = get_model(model_path,"deep")
    model = models[0]
    x_background,y,guides,otss_dict = get_data(data_path,only_seq)
    if specific_guides is None:
        specific_guides = guides
    guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    whole_background = np.concatenate(x_background)
    whole_selected = []
    for idx in guide_idx:
        sgrna = specific_guides[idx]
        sg_x_background = x_background[idx]
        sg_y = y[idx]
        sg_otss = otss_dict[sgrna]
        sg_x_selected, sgrna_otss, additional_features = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss,  only_seq, specific_indices)
        whole_selected.append(sg_x_selected)
        # NOTE: Background set is spesific to each gRNA, maybe check for equal background for all guides.
        if plot_single_guides:
            temp_output = create_folder(output_path,sgrna)
            shap_values = get_shaply_values(model, whole_background, explainer_type, sg_x_selected)
            plot_shap(shap_values, additional_features, sgrna_otss, temp_output)
    # plot all guides
    if plot_all_guides:
        whole_selected = [x[:100] for x in whole_selected] # get first 100 samples
        whole_selected = np.concatenate(whole_selected)
        shap_values = get_shaply_values(model, whole_background, explainer_type, whole_selected)
        temp_output = create_folder(output_path,"All_guides")
        plot_shap(shap_values, additional_features, None, temp_output)



def plot_shap(shap_values, additional_features, sgrna_otss, output_path):
    row_labels, x_ticks = nucleotides_for_heatmap()
    
    # Plot first 10 samples
    single_shap_values = shap_values[:10] 
    sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(single_shap_values, 24,25,additional_features)
    kwargs = {'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values'}
    plot_subplots(sequence_shap_values,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
                    y_ticks=row_labels, output_path=output_path, general_title="10-SHAP values all_bg",sgrna_otss=sgrna_otss,**kwargs)
    # Summarized plot of all samples
    summarized_shap_values = np.sum(shap_values.values, axis=0)
    sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(summarized_shap_values, 24,25,additional_features)
    kwargs = {'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values'}

    plot_subplots(sequence_shap_values,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
                    y_ticks=row_labels, output_path=output_path, general_title="Summed-SHAP values all_bg",sgrna_otss=None,**kwargs)
    # Abs mean
    abs_mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
    sequence_shap_values, epigenetic_shap_values, min_shap,max_shap  = transform_to_heatmap(abs_mean_shap_values, 24,25,additional_features)
    kwargs = {'vmin': min_shap, 'vmax': max_shap,'cbar': 'SHAP values'}
    plot_subplots(sequence_shap_values,plot_types='heatmap',titles=None, x_label="Position", y_label="Nucleotides",x_ticks=x_ticks,
                        y_ticks=row_labels, output_path=output_path, general_title="AbsMean-SHAP values all_bg",sgrna_otss=None,**kwargs)

def main_shap():
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1/model_1.keras"
    seq_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1/model_1.keras"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    
    explainer_type = "deep"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 10
    run_shap(model_path=seq_model_path,data_path=test_data_path,explainer_type=explainer_type,
             output_path=output_path,num_of_points=number_of_points,specific_guides=specific_guides,only_seq=True,plot_all_guides=False)

##################### Gradient asecnt #####################
def main_gradient_ascent():
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1/model_1.keras"
    seq_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1/model_1.keras"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 10
    run_gradient_asecnt(model_path=seq_model_path,data_path=test_data_path,output_path=output_path,
                        num_of_points=number_of_points,specific_guides=specific_guides,only_seq=True)

def run_gradient_asecnt(model_path, data_path, output_path,
             num_of_points=None,specific_indices=None, specific_guides=None,
             only_seq=False):
    from models import replace_argmax_layer
    models = get_model(model_path,"deep")
    model = models[0]
    model = replace_argmax_layer(model)
    print(model.summary())
    x = np.random.randint(2,size=(1,600),dtype=np.int8)
    alpha = 0.1
    
    # Convert input to trainable variable
    #input_var = tf.Variable(x, dtype=tf.float32)
    for i in range(10):
        grad = get_gradients(model, x)
        x += grad * alpha
    temp = x.numpy()
    seq = temp[0, :-11]
    seq = seq.reshape(32, 4)
    # Gradient ascent loop




# import logomaker
# import tensorflow as tf
# from matplotlib import pyplot as plt
# from CRISPRepi import *


def get_gradients(model, input_data):
    input_data = tf.convert_to_tensor(input_data,dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        preds = model(input_data,training=True)
    grads = tape.gradient(preds, input_data)
    
    return grads


# def saliency_map(model, epigenetics):
#     x1 = np.ones(shape=(32, 4)) * 0.25  # Initial input matrix
#     x1[21:23] = [0, 0, 1, 0]  # GG
#     x2 = np.ones(shape=(1, 11))
#     x2[:, 10] = 100  # CRISPROn
#     x2[:, 3] = avgEpi(epigenetics)  # methylation
#     x1 = x1.reshape(1, -1)
#     x = np.concatenate((x1, x2), axis=1)
#     lr = 0.1
#     # lr = 0.1, range(50000)
#     for i in range(50000):
#         grads = get_gradients(model,  x)
#         x += grads * lr
#     temp = x.numpy()
#     seq = temp[0, :-11]
#     seq = seq.reshape(32, 4)
#     seqDF = pd.DataFrame(seq, columns=['A', 'C', 'G', 'T'])
#     seqDF = seqDF.div(seqDF.sum(axis=0), axis=1)
#     save_logo(seqDF)
#     epi = x[:, -11:].numpy()
#     epi[:, 10] /= 100
#     #epi = np.array([[1.60, 1.14, 1.68, -1.65, 1.23, 1.31, -0.15, 1.44, -0.34, 0.93, 1.00]])
#     visualize_integrated_gradients(seq, 1)

#     visualize_integrated_gradients(epi, 0)


# def visualize_integrated_gradients(integrated_gradients, segFlag):
#     colors = ['white', 'white', 'white', 'black', 'white', 'white', 'black', 'white', 'black', 'white', 'white']
#     if segFlag:
#         integrated_gradients = np.flip(np.rot90(integrated_gradients, k=-1), axis=1)
#     else:
#         num_rows, num_cols = integrated_gradients.shape
#         for i in range(num_rows):
#             for j in range(num_cols):
#                 color = colors[j]
#                 plt.text(j, i, f'{integrated_gradients[i, j]:.2f}', ha='center', va='center', color=color, fontsize=10)
#     plt.imshow(integrated_gradients, cmap='Blues', interpolation='nearest', vmin=-2, vmax=2)
#     plt.colorbar(label='Feature importance')  # Add color bar
#     plt.axis('off')
#     plt.show()


# def save_logo(df):
#     IG_logo = logomaker.Logo(df)
#     IG_logo.ax.set_xticks(range(32))
#     IG_logo.ax.set_xticklabels(np.arange(1, 33), fontsize=12)
#     IG_logo.ax.set_ylabel('Importance score', fontsize=14)
#     plt.show()

# def format_titles(sgrna_otss):
#     """
#     Formats the titles for each sgRNA-OT pair to ensure alignment.
    
#     Parameters:
#         sgrna_otss (list of tuples): Each tuple contains (sgRNA, OT) sequences.
    
#     Returns:
#         list of str: Formatted titles with aligned labels.
#     """
#     labels = ["sgRNA:", "OT:"]
#     max_label_length = max(len(label) for label in labels)  # Ensures equal prefix length

#     # Generate aligned titles
#     titles = [
#         f"{'sgRNA:'.ljust(max_label_length)} {pair[0]}\n"
#         f"{'OT:'.ljust(max_label_length)} {pair[1]}"
#         for pair in sgrna_otss
#     ]

#     return titles 
    
##################### Epigenetics #####################   
def main_epigenetics():
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1/model_1.keras"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 10
    run_epigenetics(model_path=epi_model_path,data_path=test_data_path,output_path=output_path,
                    num_of_points=number_of_points,specific_guides=specific_guides)
def run_epigenetics(model_path ,data_path , output_path = None, num_of_points = 200,
                     specific_guides = None, by = None, features = None):
    '''
    This function will run epigenetic interpertation on the given model and data.
    It will interpert eather by petrubating the epigenetic features or by constant values (0.5) for each feature not evaluated.
    Args:
        model_path (str): path to the model/folder of models
        data_path (str): path to the data
        output_path (str): path to save the plots
        num_of_points (int, optional): Number of points to interpert - default 200
            None: Balanced amount of positive and negatives will be returned.
            0: all positives will be returned.
            >0: total number of points to sample.
        specific_guides (list, optional): Specific guides to extract from the data.
        by (str, optional): If 'pertubation' the epigenetic features will be pertubated, otherwise constant values will be used.
    '''
    models = get_model(model_path,"deep")
    model = models[0]
    # x_background,y,guides,otss_dict = get_data(data_path,only_seq=True)
    # if specific_guides is None:
    #     specific_guides = guides
    # guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    # # get one sample
    # sg_x, y_x, sg_ots = x_background[guide_idx[0]], y[guide_idx[0]], guides[guide_idx[0]]
    # ots_1_vector, ots_1_seq = sg_x[0], otss_dict[sg_ots][0]
    # get epigenetic features
    sg_x = np.random.randint(2, size=(1, 600), dtype=np.int8)
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    pert_import = epigenetic_pertubation_importance(features,sg_x, model)
    importance_05 = epigenetic_05_importance(features,sg_x, model)
    pert_import = {key: np.mean(value) for key, value in pert_import.items()}
    print(pert_import)
    print(importance_05)

     
if __name__ == "__main__":
    main_epigenetics()
    
