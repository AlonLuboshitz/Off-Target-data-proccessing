import pandas as pd
import numpy as np
from features_and_model_utilities import get_feature_name
from correlation_2 import hypergeometric_test, feature_correlation
from features_engineering import get_epi_data_bw


def epigenetic_enrichment_by_binary( features_columns, label_column,  data_frame):
    """
    Calculate binary feature enrichment - abundance of feature given data using the hypergeometric test.
    
    Creates a pd.DataFrame with information of the enrichment:
    
    Index:
    ['positive_peaks','negative_peaks','positive_enrichment','negative_enrichment',
    'p_val','geo_fold_pos','geo_fold_negative','positives','negatives']
    
    Columns:
    ['feature1','feature2',...]

    Args:
        features_columns (list): List of feature columns to calculate enrichment for.
        label_column (str): Column name of the label.
        data (pd.DataFrame, optional): Dataframe containing the data. If None, data_path must be provided.

    Returns:
        pd.DataFrame: Dataframe containing the enrichment results.
    


    """
    data = pd.read_csv(data_frame) if isinstance(data_frame,str) else data_frame
    positives = len(data[data[label_column] > 0])
    negatives = len(data[data[label_column] == 0])
    index = ['positive_peaks','negative_peaks','positive_enrichment','negative_enrichment','p_val','geo_fold_pos','geo_fold_negative','positives','negatives']
    enrichment_data_set = pd.DataFrame()
    enrichment_data_set["Index"] = index
    enrichment_data_set.set_index("Index",inplace=True)
    for feature in features_columns:
        feature_in_positive = len(data[(data[feature] > 0 ) & (data[label_column] > 0)])
        feature_in_nagative = len(data[(data[feature] > 0 ) & (data[label_column] == 0)])
        total_feature = feature_in_positive + feature_in_nagative
        positive_enrichment = feature_in_positive/positives
        negative_enrichment = feature_in_nagative/negatives
        p_val,geo_fold_positive,geo_fold_negative = hypergeometric_test(None,None,None,(positives,negatives,total_feature,feature_in_positive),True)
        feature_str = get_feature_name(feature)
        enrichment_data_set[feature_str] = [feature_in_positive,feature_in_nagative,positive_enrichment,negative_enrichment,p_val,geo_fold_positive,geo_fold_negative,positives,negatives]
    return enrichment_data_set


def get_sampled_coords(data, mismatch_lim = 6, mismatch_column = None, 
                       chrom_column = None, center_position_column = None, sample_size = 1000):
    """
    Returns a dictionary of sampled coordinates, chromosomes and center positions for each mismatch number.
    {missmatch_number: ([chromosomes], [center_positions])}

    Args:
        data (pd.DataFrame): Data frame with the data.
        mismatch_lim (int): Maximum number of mismatches.
        mismatch_column (str): Column name for the mismatch number.
        chrom_column (str): Column name for the chromosome.
        center_position_column (str): Column name for the center position.
        sample_size (int): Number of samples to return.
    Returns:
        dict: Dictionary of sampled coordinates, chromosomes and center positions for each mismatch number.
    """
    mismatch_groups = data.groupby(mismatch_column)
    sampled_coords = {}
    for mismatch_num, group in mismatch_groups:
        if mismatch_num > mismatch_lim or mismatch_num==0:
            continue
        sampled_group = group.sample(n=min(sample_size, len(group)), random_state=42)
        sampled_coords[mismatch_num] = (sampled_group[chrom_column].tolist(), sampled_group[center_position_column].tolist())
    return sampled_coords

def get_basepair_epigenetics_around_center(chrom_list, center_position_list,
                                            bigiwg_file, window_size, if_average=True):
    """
    Returns the epigenetic base pair values for given chromosomes and positions in a given window.

    NOTE: by defualt the epigenetic values are averaged over the window size.
    
    Args:
        chrom_list (list): List of chromosomes names.
        center_position_list (list): List of center positions.
        bigiwg_file (pybigwig object): bigwig object.
        window_size (int): Window size for averaging.
        if_average (bool defualt - True): If True, average the values over the window size.
    
    Returns:
        np.array: Epigenetic values for the given chromosomes and positions.
    """
    if len(chrom_list) != len(center_position_list):
        raise ValueError("Chromosome and center position lists must have the same length.")
    epigenetic_values = np.zeros((len(chrom_list), window_size),dtype=np.float32)
    for i, (chrom, center_position) in enumerate(zip(chrom_list, center_position_list)):
        epigenetic_values[i] = get_epi_data_bw(epigenetic_bw_file=bigiwg_file,chrom=chrom,center_loc=center_position,window_size=window_size,max_type=1)
    if if_average:
        epigenetic_values = np.mean(epigenetic_values, axis=0)
    return epigenetic_values

def get_epigentics_around_center(merged_data,on_column,label_value_list,
                                 center_value_column,chrom_column,file_manager,window_size):
    epigenetics_object = file_manager.get_bigwig_files()
    epi_dict = {}
    for epigeneitc_mark, epigenetic_file in epigenetics_object: # for each epi mark create a list with tuples - (name, average values)
        epi_dict[epigeneitc_mark] = []
        for name,label_value in label_value_list: # for each data points get averages value
            averages = average_epi_around_center(merged_data=merged_data,on_column=on_column,label_value=label_value,center_value_column=center_value_column,chrom_column=chrom_column,epigenetic_file=epigenetic_file,window_size=window_size)
            epi_dict[epigeneitc_mark].append((name,averages))
    return epi_dict