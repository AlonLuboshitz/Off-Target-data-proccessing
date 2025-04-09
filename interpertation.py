'''
Module to interpret data and models
'''
import pandas as pd
import numpy as np
import os
import shap
import Levenshtein 
import logomaker
from file_utilities import create_folder
from Data_labeling_and_processing import remove_unwanted_samples
from correlation_2 import hypergeometric_test, feature_correlation
from features_and_model_utilities import get_feature_name
from plotting import plot_correlation, plot_subplots, plot_logo
from k_groups_utilities import extract_guides_from_partition
from train_and_test_utilities import keep_intersect_guides_indices
from interpertation_utilities import *
from scipy.stats import pearsonr
from plotting_utilities import return_colormap


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
def plot_mismatch_and_bulges_disterbution(data, mismatch_column, bulges_column):
    '''
    Plot the distribution of mismatches and bulges in the data
    Args:
        data (pd.DataFrame): Dataframe containing the data.
        mismatch_column (str): Column name of the mismatches.
        bulges_column (str): Column name of the bulges.'''
    if isinstance(data,str):
        data = pd.read_csv(data)
    # only mismatches
    mismatch_data = data[data[bulges_column] == 0]
    bulges_data = data[data[bulges_column] > 0]
    if len(mismatch_data) + len(bulges_data) != len(data):
        raise ValueError("Data is not correctly seperated")
    only_mm_counts = mismatch_data[mismatch_column].value_counts()
    bulges_counts = bulges_data[bulges_column].value_counts()
    bulges_and_mismatches_counts = bulges_data[[mismatch_column,bulges_column]].value_counts()
    print(f'Only mismatches counts: {only_mm_counts}\nBulges counts: {bulges_counts}\nBulges and mismatches counts: {bulges_and_mismatches_counts}')


def create_guides_logo(output_path, counts_data = None,data_path=None, target_column=None ):
    """
    Create a logo from all the guides in the data frame.
    
    Args:
        data_path (str): Path to the data file.
        target_column (str): Column name of the target sequence.
        output_path (str): Path to save the logo.
    
    Returns:
        None
    """
    if counts_data:
        pass
    elif data_path:
        guides = pd.read_csv(data_path)[target_column].unique()
    counts_df = logomaker.alignment_to_matrix(sequences=guides, to_type='counts')
    counts_df = counts_df.drop([20,21,22])  # Remove PAM sequence
    counts_df = counts_df/len(guides)  # Normalize the counts
    plot_logo(counts_df, output_path=output_path, ax_title="Guides logo", x_label="Position", y_label="Nucleotides")

def main_logo():
    data = 'Data/Change-seq/Processed_data/78_guide_seqs.csv'
    target = 'target'
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/features'
    create_guides_logo(output_path=output_path,data_path=data,target_column=target)


def bla(bigwig_path = None, bed_path = None, amplified_data=None, vivo_data = None, vitro_data = None, silico_data = None,
        mismatch_range = 6, window_size=20000, output_path = None, mismatch_dict_sample_constraint = None):
    from file_management import File_management
    
    d = {"rHamp-seq": amplified_data,"Guide-seq": vivo_data,
                   "Change-seq": vitro_data,"Cas-offinder": silico_data}
    datas_dict = {}
    for data_name, data in d.items():
        if data is not None:
            datas_dict[data_name] = get_sampled_coords(data, mismatch_lim=mismatch_range,
                                                       mismatch_column='missmatches', chrom_column='chrom', center_position_column='chromStart')
        
    if len(datas_dict) == 0:
        raise ValueError("No data to plot")
    # for mismatch_num in range(1,mismatch_range+1): # sample data points by 
    #     sample_size = min(number_points[mismatch_num], 1000)
    #     changeseq_coords[mismatch_num] = (changeseq_coords[mismatch_num][0][:sample_size], changeseq_coords[mismatch_num][1][:sample_size])
    #     casofinder_coords[mismatch_num] = (casofinder_coords[mismatch_num][0][:sample_size], casofinder_coords[mismatch_num][1][:sample_size])
    file_manager = File_management(job='interpertation')
    
    if bigwig_path:
        file_manager.set_bigwig_folder_path(bigwig_path)
        file_manager.create_bigwig_files_objects()
        epigenetic_files = file_manager.get_bigwig_files()
        prefix = 'Bigwig'
        plot_type = 'bigwig'
        temp_path = create_folder(output_path, 'bigwig')
    elif bed_path:
        epigenetic_files = file_manager.get_bed_files()
        prefix = 'Bed'
        plot_type = 'bed'
        pass
        temp_path = create_folder(output_path, 'bed')
    else:
        raise ValueError("No bigwig or bed path given")
    
    # turn dict into mismatch num: chroms, coords
    datas_dict_by_mismatch = {mismatch_num: [data[mismatch_num] for data in datas_dict.values()] for 
                  mismatch_num in datas_dict[next(iter(datas_dict))]}
    temp_path = create_folder(temp_path,f'{window_size}_window')
    for epi_mark,epi_file in epigenetic_files:
        mark_dict = {}
    
        for mismatch_num, all_data in datas_dict_by_mismatch.items():
            chroms = [data_[0] for data_ in all_data]
            coords = [data_[1] for data_ in all_data]
            epi_peaks = [get_basepair_epigenetics_around_center(chrom, all_centers, epi_file, window_size,True) 
                         for chrom, all_centers in zip(chroms, coords)]
            
            mark_dict[mismatch_num] = {data_name: epi_peaks[i] for i, data_name in enumerate(datas_dict.keys())}
        titles = [f'{mismatch_num}_mismatch' for mismatch_num in mark_dict.keys()]
        data = [val for val in mark_dict.values()]
        kargs = {'window_size': window_size}
        general_tit = f'{prefix} {epi_mark}'
        plot_subplots(data=data,plot_types=plot_type,titles=titles,
                       general_title=general_tit,output_path=temp_path,**kargs)
def main_features_window():
    bigwig_path = 'Epigenetics/Change-seq/bigwig'
    change_seq_guide_seqs = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/78_guide_seqs.csv'
    change_seq_vitro_silico = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vitro-silico-110_withEpigenetic.csv'
    change_seq_guide_seqs = pd.read_csv(change_seq_guide_seqs)
    change_seq_vitro_silico = pd.read_csv(change_seq_vitro_silico)
    change_seqs = change_seq_vitro_silico[change_seq_vitro_silico['Label'] >0]
    casofinders = change_seq_vitro_silico[change_seq_vitro_silico['Label'] ==0]
    window_size = 100
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/features'
    bla(bigwig_path=bigwig_path,bed_path=None,amplified_data=None,vivo_data=change_seq_guide_seqs,
        vitro_data=change_seqs,silico_data=casofinders,window_size=window_size,output_path=output_path)

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
def epigenetic_importance_for_offtargets_pert_05(sg_x, features, model):
    '''
    Calculate the epigenetic importance for all off-targets both pertubation and 05 analysis.
    Args:
        sg_x (np.array): All sgRNA-OT pairs.
        features (list): List of epigenetic features.
        model (tf.keras.Model): Model to interpret.
    Returns:
        2 important lists:
        mean_pertubation_importance_list (list): List of dictionaries of mean pertubation importance for each feature.
        importance_05_list (list): List of dictionaries of 0.5 importance for each feature.
    '''
    mean_pertubation_importance_list = []
    importance_05_list = []
    # Loop over off-target vectors
    for off_target_vector in sg_x:
        # Get perturbation importance and mean values for each feature
        pertubation_importance = epigenetic_pertubation_importance(features, off_target_vector, model)
        mean_pertubation_importance = {key: np.mean(value) for key, value in pertubation_importance.items()}
        
        # Get the 05 importance values for each feature
        importance_05 = epigenetic_05_vector(features, off_target_vector, model)
        
        # Append the dictionaries to their respective lists
        mean_pertubation_importance_list.append(mean_pertubation_importance)
        importance_05_list.append(importance_05)
    return mean_pertubation_importance_list,importance_05_list

def pertubation_and_05_importance(sg_x_selected, features, model, num_of_points):
    '''
    Calculate the epigenetic importance for all off-targets both pertubation and 05 analysis.
    Args:
        sg_x_selected (np.array): Selected sgRNA-OT pairs.
        features (list): List of epigenetic features.
        model (tf.keras.Model): Model to interpret.
        num_of_points (int): Number of points that sampled.
    Returns:
        epigenetic_importance_arrays (dict): Dictionary of 2D arrays for each feature.
        1 row: pretubation.
        2 row: 0.5 importance.
    '''
    epigenetic_importance_arrays = {feature: np.zeros((2, num_of_points)) for feature in features}

    mean_pertubation_importance_list, importance_05_list = epigenetic_importance_for_offtargets_pert_05(sg_x_selected, features, model)
    epigenetic_importance_arrays = convert_importance_dicts_to_2d_arrays(epigenetic_importance_arrays,
                                                                            mean_pertubation_importance_list, importance_05_list)
    return epigenetic_importance_arrays


def run_epigenetics(model_path, features, output_path = None, guide_list = None, mismatch_limit = 3,
                    save_model_scores = True, use_model_scores = False,
                    epigenetic_disterbution_path = None):
    '''
    This function will run epigenetic interpertation on the given model and guides.
    For each guide it will create syntethic off-targets with all optional mismatches.
    Than for each off-target it will add an epigenetic vector where all the epigenetic features are 1 and 0.
    The diffrenences in model prediction for all off targets i.e. the prediction with 1 in the epigenetic feature - the prediction with 0
    in the epigenetic feature will be box plotted.
    The model predictions can be saved for further extraction and avoiding running the model again.
    The delta in predictions can be saved aswell.
    
    The function will create 2 folders:
        by_mismatch: subplot each guide with the same number of mismatches
        by_guide: subplot each mismatch number with the same guide.
    
    Args:
        model_path (str): path to the model/folder of models
        features (list): List of epigenetic features.
            The feature list should match the feature assignmnet in the model!
        output_path (str): path to save the plots, model predictions, and delta in predictions.
        guide_list (list): List of guides to use.
        mismatch_limit (int): Maximum number of mismatches to create.
        save_model_scores (bool): If True, save the model scores for each guide and mismatch.
        use_model_scores (bool): If True, use the model scores from the model_scores_path.
        epigenetic_disterbution_path (str): Path to the epigenetic distribution file.
        
    Returns:
        None
        
    '''
    def get_model_outputs_from_synthesized_ots(guide_list):
        # Create off targets
        guides_dict = create_off_targets_for_guides(guide_list=guide_list,mismatch_limit=mismatch_limit)
        # Sample data - 3,4,5,6 to many options
        
        # Create epigenetic vector and add it to the off targets
        
        
        epigenetic_disterbution_file = pd.read_csv(epigenetic_disterbution_path)
        # for clarity creating a new dictionary
        guide_dict_with_epigentics = add_epigenetic_vector_to_offtargets(guides_dict,features,epigenetic_disterbution_file)
        
        del guides_dict
        
        # Get scores
        model_outputs = get_model_scores_for_features(model_path=model_path,
                                                        guide_dict_with_epi_genetics=guide_dict_with_epigentics,
                                                        save_model_scores=save_model_scores,
                                                        output_path=output_path)
        return model_outputs
    #NOTE: add the creation of off targets that not included in the model outputs already.
    guide_list = ['GAAGGCTGAGATCCTGGAGGCGG','GAGAATCAAAATCGGTGAATCGG','GCTGGTACACGGCAGGGTCAAGG','GGACTGAGGGCCATGGACACAGG',
                  'GTCAGGGTTCTGGATATCTGTGG','GTCCCTAGTGGCCCCACTGTTGG']
    # guide_list = ['GATGCAGAGACCCTGCTCAACGG','GGGACTCTACATCTGCAAGGAGG','GCTTGTCCGTCTGGTTGCTGCGG','GCTGGCGATGCCTCGGCTGCAGG',
    #               'GCCCTGCTCGTGGTGACCGATGG','GGTGAGGGAGGAGAGATGCCTGG']
    mismatch_limit = 3
    features = [get_feature_name(feature) for feature in features]

    # Open data if given
    if use_model_scores:
        model_scores_path = os.path.join(output_path,'Model_scores','Raw_scores')
        model_outputs,guides_not_checked = open_model_outputs_file(model_scores_path, guide_list, mismatch_limit)
        if guides_not_checked is not None:
            print(f'Guides not checked: {guides_not_checked}')
            model_outputs.update(get_model_outputs_from_synthesized_ots(guides_not_checked))
    
    else:
        model_outputs = get_model_outputs_from_synthesized_ots(guide_list)
        
    # model outputs {guide: {mismatch_num: ensemble_score}}
    for guide, mismatch_dict in model_outputs.items():
        for mismatch_num, ensemble_score in mismatch_dict.items():
            model_outputs[guide][mismatch_num] = epi_feature_importance_from_model_output(features,ensemble_score)
    #save_importance_values(model_outputs, output_path)
    # importance_scores = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores/Importance_scores'
    # from_file = load_importance_values(importance_scores,guide_list,mismatch_limit)
    # Sub plot all guides togther by mismatch number.
    # Create {guide : {feature : [importance]}} by the same mismatch number
    color_map = return_colormap(features)
    
    mismatch_path = create_folder(output_path,'By_mismatch')

    for mismatch_num in range(1,mismatch_limit+1):
        model_outputs_by_mismatch = [model_outputs[guide][mismatch_num] for guide in model_outputs]
        titles = [f'{guide}' for guide in model_outputs]
        plot_epigenetic_importance_by_pertubation(model_outputs_by_mismatch,mismatch_path,colormap=color_map,title_prefix=f'{mismatch_num}_mismatch',titles=titles)
        
    # Sub plot all mismatch number by the same guide.
    guide_path = create_folder(output_path,'By_guide')
    for guide,mismatch_dict in model_outputs.items():
        data = [mismatch_dict[mismatch_num] for mismatch_num in range(1,mismatch_limit+1)]
        titles = [f'Mismatch_{mismatch_num}' for mismatch_num in range(1,mismatch_limit+1)]
        plot_epigenetic_importance_by_pertubation(data,guide_path,colormap=color_map,title_prefix=f'{guide}',titles=titles)




def old_epigenetics_function(model_path,data_path,output_path = None,
                     num_of_points = 200, specific_guides = None , features = None,
                     plot_single_guides = True, plot_all_guides = True):
    '''
    epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    test_data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    specific_guides = None
    number_of_points = 200
    features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    run_epigenetics(model_path=epi_model_path,data_path=test_data_path,output_path=output_path,
                    num_of_points=number_of_points,specific_guides=specific_guides,features=features,
                    plot_single_guides=False, plot_all_guides=True,save_model_scores=False,use_model_scores=True,
                    model_scores_path='/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores')

    '''
    
    models,models_path = get_model(model_path,"deep", sample=5)
    
    x_background,y,guides,otss_dict = get_data(data_path,only_seq=True)
    if specific_guides is None:
        specific_guides = guides
    guide_idx = keep_intersect_guides_indices(guides,specific_guides)
    
    color_map = return_colormap(features)
    titles = [f'{os.path.basename(i).split(".")[0]}' for i in models_path]
    models_results = []
    for model in models:
        whole_guides_interpertation = []
        for idx in guide_idx:
            sgrna = specific_guides[idx]
            sg_x_background = x_background[idx]
            sg_y = y[idx]
            sg_otss = otss_dict[sgrna]
            sg_x_selected, sgrna_otss, additional_features = filter_data_for_interpertation(sg_x_background, sg_y, sg_otss,
                                                                                            only_seq=True, number_of_points=num_of_points )
            #NOTE: pertubation is correlated with 0.5 therefor running only 0.5
            #epigenetic_importance_arrays = pertubation_and_05_importance(sg_x_selected, features, model, num_of_points) 
            epigenetic_importance_arrays = epigenetic_05_vector(features, sg_x_selected, model)
            whole_guides_interpertation.append(epigenetic_importance_arrays)
            if plot_single_guides:
                temp_output = create_folder(output_path,sgrna)
                plot_epigenetic_importance_by_pertubation(epigenetic_importance_arrays, temp_output, color_map)
        whole_dict = {}
        for key in whole_guides_interpertation[0]:
            arrays = [d[key] for d in whole_guides_interpertation]
            whole_dict[key] = np.hstack(arrays)
        models_results.append(whole_dict)
    if plot_all_guides:
        # whole_dict = {}
        # for key in whole_guides_interpertation[0]:
        #     arrays = [d[key] for d in whole_guides_interpertation]
        #     whole_dict[key] = np.hstack(arrays)
        temp_output = create_folder(output_path,"All_guides")
        plot_epigenetic_importance_by_pertubation(models_results, temp_output, color_map,title_prefix='Combined models', titles=titles)
        # plot_epigenetic_importance_by_pertubation(whole_guides_interpertation, temp_output, color_map,title_prefix='Separated')

        

def open_model_outputs_file(model_scores_path, guide_list, mismatch_limit,
                            model_suffix = None):
    """
    Open the model outputs from the given path.
    If some guides are not included in the model scores path, the user will be prompted to compute their scores.
    
    Args:
        model_scores_path (str): Path to the model scores - folder/guides_mismatches_scores.npy
        guide_list (list): List of guides to extract from the model scores.
        mismatch_limit (int): Maximum number of mismatches to consider.
    """
    model_outputs = {}
    if not os.path.exists(model_scores_path):
        raise ValueError("Model scores path does not exist")
    guides_in_path = [i.split("_")[0] for i in os.listdir(model_scores_path) if "scores" in i]
    difference_guides = set(guide_list).difference(set(guides_in_path))
    if len(difference_guides) > 0:
        print(f'The guides:\n{difference_guides}\nare not in the model scores path.\nWould you like compute their scores?\n1: Yes\n2: No')
        answer = input()
        if answer == "1":
            difference_guides = list(difference_guides)
        else:
            pass
    else: # Read all the model scores
        difference_guides = None
        error_file = f'{model_scores_path}/error_file.txt'
        for guide in guide_list:
            mismatch_dict = {}
            for mismatch_num in range(1,mismatch_limit+1):
                temp_guide_str = f'{guide}_{mismatch_num}_scores.npy' if not model_suffix else f'{guide}_{mismatch_num}_{model_suffix}_scores.npy'
                temp_path = os.path.join(model_scores_path,temp_guide_str) 
                if not os.path.exists(temp_path):
                    print(f'{temp_guide_str} does not exist')
                    with open(error_file,'a') as f:
                        f.write(f'{temp_guide_str}\n')
                model_scores = np.load(temp_path)
                mismatch_dict[mismatch_num] = model_scores
            model_outputs[guide] = mismatch_dict
    return model_outputs, difference_guides  

    
def plot_epigenetic_importance_by_pertubation(epigenetic_importance_arrays, output_path, colormap=None, title_prefix = "",
                                              titles = None):
    '''
    Plots the epigenetic importance pertubation calculation.
    If the epigenetic_importance_arrays is a dictionary of 2D arrays - there are 2 calculations for each feature.
        1. pretutabion. 2. 0.5 importance. 
        it will plot 2 box plots of mean pertubation values and 0.5 values and correlation between the 2 set of values.
    If the epigenetic_importance_arrays is a dictionary of 1D arrays - there is only 1 calculation for each feature.
        It will plot a box plot of the values.
    Args:
        epigenetic_importance_arrays (dict): Dictionary of 1/2D arrays for each feature.
        output_path (str): path to save the plots
        colormap (dict): Colormap for the features.
    '''
    if isinstance(epigenetic_importance_arrays,list):
        first_guide_dict = epigenetic_importance_arrays[0]
        first_key, first_value = next(iter(first_guide_dict.items()))
    elif isinstance(epigenetic_importance_arrays,dict):
        first_key, first_value = next(iter(epigenetic_importance_arrays.items()))
        if isinstance(first_value, dict): # Key: {epigenetic: [values]}
            data = [pd.DataFrame(guide_dict) for guide_dict in epigenetic_importance_arrays.values()]
            titles = [f'{guide}' for guide in epigenetic_importance_arrays.keys()]
    kwargs = {'colormap': colormap, 'showfliers': False,'showmeans':False,"order_by":"median"}
    
    if first_value.ndim == 2:
        epigenetic_importance_cor = {key: pearsonr(val[0],val[1]) for key, val in epigenetic_importance_arrays.items()}
        plot_subplots(data=epigenetic_importance_cor,plot_types='correlation', titles=None,
                    x_label='Mean pertubation importance',y_label='05 importance',
            output_path=output_path,general_title='Mean vs 0.5 importance correlation')
        pert_dict = {key: val[0] for key, val in epigenetic_importance_arrays.items()}
        dict_05 = {key: val[1] for key, val in epigenetic_importance_arrays.items()}
        data = [pd.DataFrame(pert_dict), pd.DataFrame(dict_05)]
        titles = ['Mean pertubation importance', '0.5 importance']
        plot_subplots(data=data,plot_types='boxplot',titles=titles, x_label='Epigenetic marks',y_label=f'{chr(916)} Prediction',
                output_path= output_path,general_title=f'{title_prefix} Mean vs 0.5 importance boxplot',**kwargs)   
    else: # 1D arrays
        if isinstance(epigenetic_importance_arrays,list):
            data = [pd.DataFrame(guide_dict) for guide_dict in epigenetic_importance_arrays]
        else:
            data = [pd.DataFrame(epigenetic_importance_arrays)]

        plot_subplots(data=data,plot_types='boxplot',titles=titles, x_label='Epigenetic marks',y_label=f'{chr(916)} Prediction',
                output_path= output_path,general_title=f'{title_prefix} importance boxplot',**kwargs)
 
def main_epigenetics():
    #seq_path='/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Only_sequence/All_guides/1_ensembels/50_models/ensemble_1'
    all_epi_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/All-epigenetics/ensemble_1"
    all_epi_features = ["H3K27me3_peaks_binary", "H3K27ac_peaks_binary", "H3K9ac_peaks_binary", "H3K9me3_peaks_binary", "H3K36me3_peaks_binary", "ATAC-seq_peaks_binary", "H3K4me3_peaks_binary", "H3K4me1_peaks_binary"]
    all_model_score_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability/Model_scores/Raw_scores'
    h3k27ac_model_path = "/localdata/alon/Models/Change-seq/vivo-silico/Exclude_Refined_TrueOT/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/With_features_by_columns/All_guides/1_ensembels/50_models/Binary_epigenetics/H3K27ac/ensemble_1"
    h3k27ac_features = ["H3K27ac_peaks_binary"]
    epi_model_path = all_epi_model_path
    features = all_epi_features
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/vivo-silico/Exclude_Refined_TrueOT/on_Refined_TrueOT_Lazzarroto/Classification/No_constraints/Full_encoding/No_CW/GRU-EMB/5epochs_1024_batch/Early_stop/Ensemble/Model_interpertability'
    model_name = epi_model_path.split('Binary_epigenetics')[1].split('/')[1] # get epigenetics used in the model
    output_path = create_folder(output_path,model_name)
    epi_dis_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/Change-seq/Epigenetic_disterbution.csv'
    run_epigenetics(model_path=epi_model_path,output_path=output_path,
                    features=features, save_model_scores=True,use_model_scores=True,
                    epigenetic_disterbution_path=epi_dis_path)
if __name__ == "__main__":
    main_logo()
    
