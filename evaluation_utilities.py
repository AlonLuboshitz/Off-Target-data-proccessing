from collections import defaultdict
import numpy as np
import pandas as pd
from utilities import extract_scores_labels_indexes_from_files
from multiprocessing import Pool
import os
from file_utilities import create_paths

def merge_by_mismatches(guides_dict, error_file):
    """
    Given a dicionary of {mismatch: {guide: {feature: (y_scores, y_test, indexes)}}}
    merge the scores,tests, indexes of each guide for each mismatch.
    
    Args:
        guides_dict (dict): Dictionary containing the guides data.
        error_file (str): Path to the error file to log issues.
    
    Returns:
        dict: Merged dictionary with the structure {mismatch: {feature: (y_scores, y_test, indexes)}}
    """
    merged_guide_dict = {}
    for mismatch, guide_dict in guides_dict.items():
        merged_guide_dict[mismatch] = defaultdict(lambda: [[], [], []])
        for guide, features in guide_dict.items():
            for feature, (y_scores, y_test, indexes) in features.items():
                merged_guide_dict[mismatch][feature][0].append(y_scores)
                merged_guide_dict[mismatch][feature][1].append(y_test)
                merged_guide_dict[mismatch][feature][2].append(indexes)
        for feature, (y_scores, y_test, indexes) in merged_guide_dict[mismatch].items():
            merged_guide_dict[mismatch][feature][0] = np.concatenate(y_scores, axis=0)
            merged_guide_dict[mismatch][feature][1] = np.concatenate(y_test, axis=0)
            merged_guide_dict[mismatch][feature][2] = np.concatenate(indexes, axis=0)
        test_samples = next(iter(merged_guide_dict[mismatch].values()))[1]
        if sum(test_samples>0) == 0:
            with open(error_file, "a") as f:
                f.write(f"all guides have no positives for mismatch {mismatch}\n")
            merged_guide_dict.pop(mismatch)
            continue
    return merged_guide_dict


def get_guide_information(data_name, guide, statistics_file):
    """
    Returns the information of a single/multiple guide/s from the statistics file.

    Args:
        data_name (str): Name of the dataset.
        guide (str or list): Guide sequence or list of guide sequences.
        statistics_file (str): Path to the statistics file.

    Returns:
        dict: Dictionary containing guide information {guide: {feature: value}}
        for feature in data.

    """
    data = pd.read_csv(statistics_file)
    data = data[data['Data_set'] == data_name] # keep corresponding data
    # if multiple guides
    keys  = ["Data_set", "Gene_name", "guide_sequence", "amplified_otss", "vivo_otss", "potential_otss"]
    if isinstance(guide, list):
        guide_info = data[data['guide_sequence'].isin(guide)]
        guide_info = guide_info.to_dict(orient='list')
        guide_info["Data_set"] = set(guide_info["Data_set"])
        guide_info["guides_amount"] = len(guide)
        guide_info["amplified_otss"] = int(sum(guide_info["amplified_otss"]))
        guide_info["vivo_otss"] = int(sum(guide_info["vivo_otss"]))
        guide_info["potential_otss"] = int(sum(guide_info["potential_otss"]))
        keys.append("guides_amount")
        keys.remove("guide_sequence")
        keys.remove("Gene_name")
    else:
        guide_info = data[data['guide_sequence'] == guide]
        guide_info = guide_info.to_dict(orient='records')[0]
    guide_info = {key: guide_info[key] for key in keys}
    return guide_info

def keep_indexes_from_scores_labels_indexes(y_scores, y_test, indexes, spesific_indexes):
    """
    Get spesific indexes from the y_scores, y_test and indexes.

    Args:
        y_scores (np.array): The prediction scores.
        y_test (np.array): The true labels.
        indexes (np.array): The indexes of the samples.
        spesific_indexes (list): The spesific indexes to keep.
    Returns:
        selected_y_scores (np.array): The selected y_scores.
        selected_y_test (np.array): The selected y_test.
        spesific_indexes (list): The spesific indexes to keep.
    """
    positional_indexes = np.where(np.isin(indexes, spesific_indexes))[0] # get the postional indexes of the spesific indexes
    if y_scores.ndim == 1:
        selected_y_scores = y_scores[positional_indexes]
    elif y_scores.ndim == 2:
        selected_y_scores = y_scores[:, positional_indexes]
    
    selected_y_test = y_test[positional_indexes]
    return selected_y_scores, selected_y_test, spesific_indexes

def init_feature_dict_for_all_scores(ml_results_paths,n_ensebmles,  reg_classification = False,
                                     additional_data = None):
    """

    This function will return a dictionary with all the features and their scores, labels and indexes.
    
    Args:
        ml_results_paths (list): list of paths to the models results folders.
            Each folder is a different model.
        n_ensebmles (int): number of ensembles in the results.
        reg_classification (bool): if the task is classification by regression.
            If True, the function will transform the labels to binary labels.
        additional_data (tuple): (name, path) - tuple of additional data to add to the feature dict.
    
    Returns:
        features_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
    
    """
    
    fill_feature_dict_args = []
    for ml_results_path in ml_results_paths:
        if "Only_sequence" in ml_results_path:
            feature = "Only-seq"
        else : 
            feature = ml_results_path.split("/")[-1]
        fill_feature_dict_args.append(({},feature,ml_results_path,n_ensebmles,reg_classification))
    proccess = min(os.cpu_count(), len(fill_feature_dict_args))
    features_dict = {}
    if additional_data:
        ##NOTE: validate path and feature
        ml_path = additional_data[0]
        feature_name = additional_data[1]
        additional_data_ = fill_feature_dict_with_scores({},feature_name,ml_path,n_ensebmles,reg_classification)
        features_dict.update(additional_data_)
    with Pool(proccess) as pool:
        results = pool.starmap(fill_feature_dict_with_scores, fill_feature_dict_args)
    
    for result in results:
        features_dict.update(result)
    return features_dict

def get_ots_indexes_till_last_tp(guides_results_dict,guides_dict):
    """
    Saves all the off-targets indexes before and including the last true-positive.

    Returns:
        Dictionary : {guide_seqeunce : np.array (2,number of points till last tp) - with the indexes and the scores}
    """
    guide_indexes_till_last_tp = {}
    for guide_seq, guide_data in guides_dict.items():
        guide_scores,guide_indexes= guide_data['Only-seq'][0],guide_data['Only-seq'][2]
        guide_only_seq_index = guides_results_dict[guide_seq][1].index('Only-seq')
        guide_last_tp = guides_results_dict[guide_seq][0]['last_fn_values'][guide_only_seq_index][0]
        array_to_save = np.zeros(shape=(2,guide_last_tp))
        sorted_predictions = np.argsort(guide_scores)[::-1]
        prediction_values = guide_scores[sorted_predictions][:guide_last_tp]
        all_indexes_till_last_tp = guide_indexes[sorted_predictions[:guide_last_tp]]
        array_to_save[0] = all_indexes_till_last_tp
        array_to_save[1] = prediction_values
        guide_indexes_till_last_tp[guide_seq] = array_to_save
    return guide_indexes_till_last_tp

def fill_feature_dict_with_scores(feature_dict, feature, scores_folder_path, n_ensembles,
                                   reg_classification = False):
    """
    For each models in the scores folder path, it will extract the scores, labels and indexes.
    Each ensemble is being averaged and the results are being saved in the feature dict.
    
    Args:
        feature_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
        feature (str): name of the feature.
        scores_folder_path (str): path to the scores folder.
        n_ensembles (int): number of ensembles in the results.
        reg_classification (bool): if the task is classification by regression.
            If True, the function will transform the labels to binary labels.
    Returns:
        feature_dict (dict): dictionary with the features and their scores, labels and indexes.
            Dictionary structure: {feature: (y_scores, y_test, indexes)}
            where y_scores is a 2d array of prediction_scores if more than 1 ensemble.
    """
    if n_ensembles > 1: # multiple ensembles in the results
        ensembles = create_paths(os.path.join(scores_folder_path, "Scores"))
        y_scores = []
        for ensemble in ensembles:
            ensemble_scores, y_test, indexes = extract_scores_labels_indexes_from_files([ensemble])
            ensemble_scores = np.mean(ensemble_scores, axis = 0)
            y_scores.append(ensemble_scores)
        y_scores = np.array(y_scores)
    else: # one ensemble
        y_scores, y_test, indexes = extract_scores_labels_indexes_from_files(create_paths(os.path.join(scores_folder_path, "Scores")))
        y_scores = np.mean(y_scores, axis = 0)
    if reg_classification: # transform y to binary labels
        y_test = (y_test > 0).astype(int)
    feature_dict[feature] = (y_scores, y_test, indexes)
    return feature_dict

def split_feature_dict_by_indexes(features_dict, indexes_dict, by_mismatch = False):
    """
    Split the features dict {feature: (scores,test,all samples indexes)} by given indexes.
    By defualt it will split by guide indexes.
    If by_mismatch is True, it will split by mismatch indexes.

    Args:
        features_dict (dict): {feature: (scores,test,all samples indexes)}
            dictionary with the features and their scores, labels and indexes.
        indexes_dict (dict): {guide: indexes} - dictionary with the sample indexes of each guide.
        by_mismatch (bool): if to split by mismatch indexes.
            if True, indexes_dict should have:
            {guide: {mismatch: indexes}} - dictionary with the sample indexes of each guide.
     
    Returns: 
        dictionary {guide: {feature : (y_scores, y_test)}}
        if by_mismatch is True, it will return: {mismatch: {guide: {feature : (y_scores, y_test)}}}
    """
    if by_mismatch:
        mismatch_dict = {}
        for mismatch_number, guide_indexes in indexes_dict.items():
            mismatch_dict[mismatch_number] = {}
            for guide, indexes in guide_indexes.items():
                mismatch_dict[mismatch_number][guide] = {}
                for feature, (y_scores, y_test, all_indexes) in features_dict.items():
                    y_indexed_scores, y_indexed_test, indexes = keep_indexes_from_scores_labels_indexes(y_scores, y_test, all_indexes, indexes)
                    mismatch_dict[mismatch_number][guide][feature] = y_indexed_scores, y_indexed_test, indexes
        return mismatch_dict
    guides_dict = {}
    for guide,indexes in indexes_dict.items():
        guides_dict[guide] = {}
        for feature, (y_scores, y_test, all_indexes) in features_dict.items():
            y_indexed_scores, y_indexed_test, indexes = keep_indexes_from_scores_labels_indexes(y_scores, y_test, all_indexes, indexes)
            guides_dict[guide][feature] = y_indexed_scores, y_indexed_test, indexes
    return guides_dict