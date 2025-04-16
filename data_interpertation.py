import Levenshtein 
import logomaker
import pandas as pd
import os
import numpy as np
from file_utilities import create_folder, create_paths
from Data_labeling_and_processing import remove_unwanted_samples
from data_interpertation_utilities import *
from plotting import plot_correlation, plot_binary_feature_heatmap, plot_logo, plot_subplots
from k_groups_utilities import extract_guides_from_partition

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



def binary_feature_enrichment_by_partition(data, features_columns, label_column,
                                           partitions = [], output_path = None,
                                            partition_info_path = None, plot = True):
    """
    Calculate the epigenetic feature enrichment for each partition and and for the whole data.
    Saves the enrichment results in csv.
    Plots the enrichment results for each partition and for the whole data.

    Args:
        data (pd.DataFrame, str): Dataframe/path containing the data.
        partitions (list): List of partitions ints to calculate the enrichment for.
        features_columns (list): List of columns with the features.
        label_column (str): Column with the label.
        output_path (str): Path to save the results.
        partition_info_path (str): Path to the partition info file.
        plot (bool): If True, plot the enrichment results.
    Returns:
        None
        
    Usage:
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
    
    """
    data = pd.read_csv(data) if isinstance(data, str) else data 
    
    if partition_info_path is None:
        raise RuntimeError("No parition data path is given")
    partition_info = pd.read_csv(partition_info_path)
    if output_path:
        output_path = create_folder(output_path, "Binary")
    results_path = create_folder(output_path, "Results")
    if plot:
        plot_path = create_folder(output_path, "Plots")

    for partition in partitions:
        partition_guides = extract_guides_from_partition(partition_info,partition)
        partition_data = data[data["target"].isin(partition_guides)]
        temp_path = os.path.join(results_path,f"{partition}_partition.csv")
        enrichment_results = epigenetic_enrichment_by_binary(features_columns,label_column,partition_data)
        enrichment_results.to_csv(temp_path)
        if plot:
            title = f"Partition {partition}"
            plot_binary_feature_heatmap(df = enrichment_results, axes = None, title = title, plots_path = plot_path)
    temp_path = os.path.join(results_path,"All_partitions.csv")
    enrichment_results = epigenetic_enrichment_by_binary(features_columns,label_column,data_frame=data)
    enrichment_results.to_csv(temp_path)
    if plot:
        title = "All partitions"
        plot_binary_feature_heatmap(df = enrichment_results, axes = None, title = title, plots_path = plot_path)

def plot_epigenetic_binary_enrichment(folder_path):
    """
    Plot the epigenetic binary enrichment for each partition and for the whole data
    Given folder with csv files containing the enrichment information.
    """
    datas = create_paths(folder_path)
    titles = [data.split('/')[-1].replace('.csv','').replace('_',' ') for data in datas]
    datas = [pd.read_csv(data, index_col=0) for data in datas]
    plots_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/features/vivo-silico/exclue_new_guides/Binary/Plots'
    for data,title in zip(datas,titles):
        plot_binary_feature_heatmap(df = data, axes = None, title = title, plots_path = plots_path)   

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


def epigenetic_enrichment_by_window(bigwig_path = None,  amplified_data=None, 
                                       vivo_data = None, vitro_data = None, silico_data = None,
                                       mismatch_range = 6, window_size=20000, output_path = None,
                                       by_scalar = False):
    """
    Calculate the epigenetic enrichment for the diffrenet off-target data types.
    Data type refer to the off-target assay, i.e. in vivo, in silico and so on.
    The function extracts off-targets coordinates for each data type.
    For these coordinates it calculates the epigenetic enrichment.

    The epigenetic data can be per base pair - bigwig and than the function will plot the average
    values of a given window size around the off-target coordinates.
    
    The epigenetic data can be scalar per off-target. 
    Two options:
        1. bed file - the function will intersect each bedfile with the off-target coordinates
        it will calculate the perctanges of the epigenetic mark for each data type.
        eather normal percantage or hypergeometric test.

        2. big wig scalar - evaluate the correlation of the avarage basepair values around the off-target

    """
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
    
    file_manager = File_management(job='interpertation')

    file_manager.set_bigwig_folder_path(bigwig_path)
    file_manager.create_bigwig_files_objects()
    epigenetic_files = file_manager.get_bigwig_files()
    prefix = 'Bigwig'
    plot_type = 'bigwig'
    temp_path = create_folder(output_path, 'bigwig')
    
    
    # turn dict into mismatch num: chroms, coords
    datas_dict_by_mismatch = {mismatch_num: [data[mismatch_num] for data in datas_dict.values()] for 
                  mismatch_num in datas_dict[next(iter(datas_dict))]}

    temp_path = create_folder(temp_path,f'{window_size}_window')

    for epi_mark,epi_file in epigenetic_files: 
        mark_dict = {} # init empty dict to hold enrichment values per epigenetic mark
        for mismatch_num, all_data in datas_dict_by_mismatch.items(): # all_data is a list of tuples (chrom, coords)
            chroms = [data_[0] for data_ in all_data]
            coords = [data_[1] for data_ in all_data]
            # for bigwig files.
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
    change_seq_vivo_silico = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vivo-silico-78_withEpigenetic.csv'
    change_seq_vitro_silico = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vitro-silico-110_withEpigenetic.csv'
    change_seq_vivo_silico = pd.read_csv(change_seq_guide_seqs)
    change_seq_vitro_silico = pd.read_csv(change_seq_vitro_silico)
    # keep only vivo guides
    vivo_guides = change_seq_vivo_silico['target'].unique
    change_seq_vitro_silico = change_seq_vitro_silico[change_seq_vitro_silico['target'].isin(vivo_guides)]
    # split to data types
    change_seq_guide_seqs = change_seq_vivo_silico[change_seq_vivo_silico['Label']>0]
    change_seqs = change_seq_vitro_silico[change_seq_vitro_silico['Label'] >0]
    casofinders = change_seq_vitro_silico[change_seq_vitro_silico['Label'] ==0]
    window_size = 100
    output_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/features'
    epigenetic_enrichment_by_window(bigwig_path=bigwig_path,amplified_data=None,vivo_data=change_seq_guide_seqs,
        vitro_data=change_seqs,silico_data=casofinders,window_size=window_size,output_path=output_path)

def main_binary_features():
    features = [
        'H3K27me3_peaks_binary', 
        'H3K27ac_peaks_binary', 
        'H3K9ac_peaks_binary', 
        'H3K9me3_peaks_binary', 
        'H3K36me3_peaks_binary', 
        'ATAC-seq_peaks_binary', 
        'H3K4me3_peaks_binary', 
        'H3K4me1_peaks_binary'
    
]
    output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Change-seq/features/vivo-silico/exclue_new_guides"
    data_path ="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vivo-silico-78_withEpigenetic.csv"
    partition_info = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Ofir_partition/vivo/Change-seq_exclude_new_guides_vivo_silico.csv"
    partitions = [i for i in range(1,11)]

    binary_feature_enrichment_by_partition(data=data_path, features_columns=features,label_column='Label',
                                           partitions=partitions,output_path=output_path,partition_info_path=partition_info)

def main_data():
    data_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
    data_frame = pd.read_csv(data_path)
    data_frame = data_frame[data_frame['target'].isin(specific_guides)]
    get_number_of_mismatches_per_position(data_frame, COLUMNS["REALIGNED_COLUMN"], COLUMNS["OFFTARGET_COLUMN"])