import numpy as np
import pandas as pd
from itertools import combinations
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from utilities import create_paths, find_target_folders,get_X_random_indices, extract_scores_labels_indexes_from_files
from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std,plot_roc, plot_correlation, plot_pr
from ml_statistics import get_ensmbels_stats, get_mean_std_from_ensmbel_results, pearson_correlation, spearman_correlation


def get_tpr_by_n_expriments(predicted_vals,y_test,n):
    '''This function gets the true positive rate for n expriemnets by calculating:
for each 1 <= n' <= n prediction values, what the % of positive predcition out of the the whole TP amount.
for example: '''
    # valid that test amount is more then n
    if n > len(y_test):
        print(f"n expriments: {n} is bigger then data points amount: {len(y_test)}, n set to data points")
        n = len(y_test)
    
    tp_amount = np.count_nonzero(y_test) # get tp amount
    if predicted_vals.ndim > 1:
        predicted_vals = predicted_vals.ravel()
    sorted_indices = np.argsort(predicted_vals)[::-1] # Get the indices that would sort the prediction values array in descending order    
    tp_amount_by_prediction = 0 # set tp amount by prediction
    tpr_array = np.empty(0) # array for tpr
    for i in range(n):
        # y_test has label of 1\0 if 1 adds it to tp_amount
        tp_amount_by_prediction = tp_amount_by_prediction + y_test[sorted_indices[i]]
        tp_rate = tp_amount_by_prediction / tp_amount
        tpr_array= np.append(tpr_array,tp_rate)
        if tp_rate == 1.0:
            # tp amount == tp amount in prediction no need more expriments and all tp are found
            tpr_array = np.concatenate((tpr_array, np.ones(n - (i + 1)))) # fill the tpr array with 1 
            break    
    return tpr_array  

def get_auc_by_tpr(tpr_arr):
    amount_of_points = len(tpr_arr)
    x_values = np.arange(1, amount_of_points + 1) # x values by lenght of tpr_array
    calculated_auc = auc(x_values,tpr_arr)
    calculated_auc = calculated_auc / amount_of_points # normalizied auc
    return calculated_auc,amount_of_points



def evaluate_model( y_test, y_pos_scores_probs):
    # # Calculate AUROC,AUPRC
    fpr, tpr, tresholds = roc_curve(y_test, y_pos_scores_probs)
    auroc = auc(fpr, tpr)
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pos_scores_probs)
    return (auroc,auprc)

def plot_roc_pr_for_ensmble_by_paths(score_paths, titles, output_path, plot_title):
    '''This function plots multiple rocs and pr curves togther for multiple models.
    It iterates the score paths given in the score paths list and plots the roc/pr curve for each model.
    The titles list should contain the title for each model.
    Args:
    1. score_paths - list of paths to the scores files.
    2. titles - list of titles for each model.
    3. output_path - path to save the plot.
    4. plot_title - title for the plot.
    -----------
    Returns: None
    Example: 
    scores_path = ["/localdata/alon/ML_results/Hendel/vivo-silico/test_on_changeseq/6_intersect/all_6/Scores/ensemble_1.csv",
                   "/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/test_on_hendel/6_intersect/all_6/Scores/ensemble_1.csv"]
    titles = ["Hendel on CHANGE-seq","CHANGE-seq on Hendel"]
    plot_roc_pr_for_ensmble_by_paths(scores_path,titles,"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq","Models_6_intersect")'''
    if len(score_paths) != len(titles):
        raise ValueError("The amount of score paths should be equal to the amount of titles")
    fprs = []
    tprs = []
    aucs = []
    percs = []
    auprcs = []
    recalls = []
    for test_path in score_paths:
        y_scores, y_test, indexes = extract_scores_labels_indexes_from_files([test_path])
        y_scores = np.mean(y_scores, axis = 0)
        fpr, tpr, tresholds = roc_curve(y_test, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc(fpr, tpr))
        percs.append(precision)
        recalls.append(recall)
        auprcs.append((average_precision_score(y_test, y_scores),np.sum(y_test[y_test > 0]) / len(y_test)))
    plot_roc(fprs,tprs,aucs,titles,output_path,f'{plot_title}_roc')
    plot_pr(recall_list=recalls,precision_list=percs,auprcs=auprcs,titles=titles,output_path=output_path,general_title=f'{plot_title}_pr')
def evaluate_n_combinatorical_models(n_models, n_y_scores, y_test, k):
    '''This function aasses all the possible combinatorical options for k given.
N choose K options.
For exmaple: if there are 10 models and n = 3, the function will average 
the results of all the possible combinations of 3 models out of 10.
The function will return the average of the aurpc,auroc and std over all the combinations.'''
    # Get list of tuples containing k indices out of n_models
    if n_models <= 10 or k==2: # more then 10 models to many combinations to validate
        indices_combinations = list(combinations(range(n_models), k))
        if len(indices_combinations) > 200: 
            indices_combinations = get_X_random_indices(n_models, k, 200)
        indices_combinations = [list(indices) for indices in indices_combinations]
    else :
        indices_combinations = get_X_random_indices(n_models, k, 200)    
    
    # Create np array for each indice combination average auroc,auprc,n_rank
    # Row - combination, Column - auroc,auprc,n_rank
    all_combination_results = np.zeros(shape=(len(indices_combinations),3))
    for index,indices in enumerate(indices_combinations):
        # get the scores for the models in the combination
        y_scores = n_y_scores[indices]
        # average the scores
        y_scores = np.mean(y_scores, axis = 0)
        # evaluate the model
        auroc, auprc = evaluate_model(y_test, y_scores)
        n_rank = get_auc_by_tpr(get_tpr_by_n_expriments(y_scores,y_test,1000))[0]
        all_combination_results[index] = [auroc, auprc, n_rank]
    return all_combination_results

def eval_all_combinatorical_ensmbel(y_scores, y_test, header = ["Auroc","Auprc","N-rank","Auroc_std","Auprc_std","N-rank_std"]):
    '''Evaluate all the possible combinatorical options for an ensmbel
y_scores is 2d array (N_models, scores), y_test is the accautal labels'''
    # Get amount of models
    n_models = y_scores.shape[0]
    # Create nd array with for each k combination with auroc,auprc,n-rank (means,std)
    all_combination_results = np.zeros(shape=(n_models,len(header)))
    # first row k = 1 no ensmble to calculate
    for k in range(1,n_models): # 1 
        print(f"Check combinations of {k + 1} models out of {n_models}")
        k_combination_result = evaluate_n_combinatorical_models(n_models, y_scores, y_test, k + 1) 
        # Average the k_combination_results over the 2d dimension
        k_combination_result_mean = np.mean(k_combination_result, axis = 0)
        k_combination_result_std = np.std(k_combination_result, axis = 0)
        k_mean_and_std = np.concatenate((k_combination_result_mean, k_combination_result_std))
        # Add the results to all_combination_results
        all_combination_results[k] = k_mean_and_std
    return all_combination_results



def extract_combinatorical_results(ensmbel_combi_path, n_models_in_ensmbel_list):
    '''Given a list of paths for csv files containing ensembel combinatiorical results
And given a list with model amounts in each ensmbel to check return a dictionary
with the results of each combinatorical amount from each ensmbel
Dict returned : {n_keys : (n_ensmbels,[auroc,auprc,n-rank]) }'''
    ensmbels_combi_paths = create_paths(ensmbel_combi_path) 
    # Create a dictinoary of 2d np arrays. key is number of models in the ensmbel and value is the results
    all_n_models = {n_models: np.zeros(shape=(len(ensmbels_combi_paths),3)) for n_models in n_models_in_ensmbel_list}
    
    for idx,ensmbel in enumerate(ensmbels_combi_paths):
        results = np.genfromtxt(ensmbel, delimiter=',') # read file
        for n_models in n_models_in_ensmbel_list:
            # n_models is the row number first row is header
            n_results = results[n_models,:3] # 3 columns - auroc,auprc,n-rank
            all_n_models[n_models][idx] = n_results 

    return all_n_models
def eval_ensembles_in_folder(ensmbel_folder_path, n_models_in_ensmbel, if_single_amount_of_models = False):
    '''Given a path to a folder containing multiple folders of models
    Iterate over each folder and return the results of each ensmbel
    n_models_in_ensmbel is a number of models in each ensmbel.
    If if_single_amount_of_models is True, the calculation will be only for the given amount of models
    other wise it will be for all the possible combinations of models in the ensmbel
    -----
    returns a dict {key : folder_name, val: dict{key: n_models_in_ensmbel, val: np array of results}}'''
    if not if_single_amount_of_models:
        n_models_in_ensmbel = [i for i in range(2,n_models_in_ensmbel+1)]
    else :
        n_models_in_ensmbel = [n_models_in_ensmbel]
    ensmbel_results = {}
    for ensmbel_folder in os.listdir(ensmbel_folder_path):
        ensmbel_combi_path = os.path.join(ensmbel_folder_path,ensmbel_folder,"Combi") # combine folder with combi results
        dict_ensmbel = extract_combinatorical_results(ensmbel_combi_path,n_models_in_ensmbel) # get the results
        folder_suffix = ensmbel_folder.split("_") # get the folder name
        if folder_suffix[0] == 'Chromstate':
            folder_suffix = folder_suffix[1]
        else:
            folder_suffix = ensmbel_folder
        ensmbel_results[folder_suffix] = dict_ensmbel # set the results per folder
    return ensmbel_results

    
def plot_all_ensmbels_means_std(ensmbel_mean_std_dict,features,stats_dict,output_path, n_modles_in_ensmbel = None):
    '''Given a dictionary with the mean and std for each ensmbel - [0] -auroc,[1] - auprc,[2] - n-rank
    plot the results for all ensmbels togther
    stats dict is {Key: group/model, value: pvals of [auroc,auprc,n-rank]}'''
    x_values = [str(ensmbel) for ensmbel in ensmbel_mean_std_dict.keys()]
    aurocs_results = [results[n_modles_in_ensmbel][0][0] for results in ensmbel_mean_std_dict.values()]
    aurocs_stds = [results[n_modles_in_ensmbel][1][0] for results in ensmbel_mean_std_dict.values()]
    roc_pvals = {key: stats_dict[key][0] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(aurocs_results,aurocs_stds,x_values,roc_pvals,f"AUROC by ensmbels - {features}","AUROC",output_path)
    auprcs_results = [results[n_modles_in_ensmbel][0][1] for results in ensmbel_mean_std_dict.values()]
    auprcs_stds = [results[n_modles_in_ensmbel][1][1] for results in ensmbel_mean_std_dict.values()]
    prc_pvals = {key: stats_dict[key][1] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(auprcs_results,auprcs_stds,x_values,prc_pvals,f"AUPRC by ensmbels - {features}","AUPRC",output_path)
    n_ranks_results = [results[n_modles_in_ensmbel][0][2] for results in ensmbel_mean_std_dict.values()]
    n_ranks_stds = [results[n_modles_in_ensmbel][1][2] for results in ensmbel_mean_std_dict.values()]
    n_rank_pvals = {key: stats_dict[key][2] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(n_ranks_results,n_ranks_stds,x_values,n_rank_pvals,f"N-rank by ensmbels - {features}","N-rank",output_path)

def plot_all_ensmbels_std(ensmbel_mean_std_dict,features):
    x_values = [str(ensmbel) for ensmbel in ensmbel_mean_std_dict.keys()]
    aurocs_stds = [results[ensmbel][1][0] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(aurocs_stds,x_values,f"AUROC stds by ensmbels - {features}","AUROC",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    auprcs_stds = [results[ensmbel][1][1] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(auprcs_stds,x_values,f"AUPRC stds by ensmbels - {features}","AUPRC",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    n_ranks_stds = [results[ensmbel][1][2] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(n_ranks_stds,x_values,f"N-rank stds by ensmbels - {features}","N-rank",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")

def get_subset_dict(subsets_path, feature_dict, n_models_in_ensmbel):
    picked_dict = eval_ensembles_in_folder(subsets_path,n_models_in_ensmbel,True)
    picked_dict['Only-seq'] = feature_dict["Only-seq"]
    picked_dict['All'] = feature_dict["All"]
    temp_keys = ''
    for keys in picked_dict.keys():
        if len(temp_keys) < len(keys):
            temp_keys = keys
    temp_keys = temp_keys.split("_")
    for key in temp_keys:
        picked_dict[key] = feature_dict[key]
    return picked_dict

def bar_plot_ensembels_feature_performance(only_seq_combi_path, epigenetics_path, n_models_in_ensmbel, output_path, title,  add_subsets = None):
    '''This function will plot the performance of the ensembles for the given features compared with only sequence
    Will create 3 diffrenet plots for AUROC,AUPRC,N-rank each with the performance of the ensembles for the given features
    The bar plots will be created and statistical test between only sequence and the other features will be calculated
    p value marks - *,**,*** will be added to the plot
    ------------
    If add subsets is given - subsets - path for subsets folders , the function will add the performance of the subsets
    and each single feature in the subset to the plot
    '''
    feature_dict = eval_ensembles_in_folder(epigenetics_path, n_models_in_ensmbel, True)
    dict_50_only_seq = extract_combinatorical_results(only_seq_combi_path, [n_models_in_ensmbel])
    feature_dict["Only-seq"] = dict_50_only_seq
    if add_subsets:
        subsets_dict = get_subset_dict(add_subsets, feature_dict, n_models_in_ensmbel)
        subsets_stats = get_ensmbels_stats(subsets_dict,n_models_in_ensmbel)
        subsets_scores = get_mean_std_from_ensmbel_results(subsets_dict)
        plot_all_ensmbels_means_std(subsets_scores,f"{title}Subsets",subsets_stats, output_path, n_models_in_ensmbel)
    feature_stats = get_ensmbels_stats(feature_dict,50)
    feature_scores = get_mean_std_from_ensmbel_results(feature_dict)
    plot_all_ensmbels_means_std(feature_scores,title, feature_stats, output_path, n_models_in_ensmbel)

def performance_by_data_points(base_path, n_models_in_ensmbel,output_path, data_name):
    '''Given a base path with the results of the ensembles for different amount of data points
    Each amount of data_points is in a different folder
    Retrive the combi results from each folder - combine the combi results from each folder and calculate
    Mean and STD for each amount of data points
    Plot each amount of data points and the mean performance with std'''
    group_dict = {}
    num_of_groups = len(os.listdir(base_path))
    for group in range(1,num_of_groups+1):
        
        group_key = f'{group}_part'
        if group == num_of_groups:
            group_key = 'All'
        group_paths = find_target_folders(os.path.join(base_path,f'{group}_group'),["Combi"])
        group_paths.sort() # sort by partition number
        group_paths = [os.path.join(path,"Combi") for path in group_paths] # Add Combi folder to each path
        partition_num = len(group_paths)
        # set array for the values - auroc,auprc,n-rank for each partition in the group
        values_arr = np.zeros(shape=(partition_num,3)) 
        for partition,path in enumerate(group_paths):
            values_arr[partition] = extract_combinatorical_results(path,[n_models_in_ensmbel])[n_models_in_ensmbel]
        group_dict[group_key] = values_arr.mean(axis=0),values_arr.std(axis=0)  
    # Sort dict by keys
    # group_dict = dict(sorted(group_dict.items()))   
    x_vals = [key for key in group_dict.keys()]
    y_vals = [value[0]  for value in group_dict.values()] 
    y_stds = [value[1] for value in group_dict.values()]
    plot_ensemeble_preformance(y_values=[val[0] for val in y_vals],x_values=x_vals,title=f"Performance by data points AUROC {data_name}",y_label="AUROC",x_label="Data points",stds=[val[0] for val in y_stds],output_path=output_path)
    plot_ensemeble_preformance(y_values=[val[1] for val in y_vals],x_values=x_vals,title=f"Performance by data points AUPRC {data_name}",y_label="AUPRC",x_label="Data points",stds=[val[1] for val in y_stds],output_path=output_path)
    plot_ensemeble_preformance(y_values=[val[2] for val in y_vals],x_values=x_vals,title=f"Performance by data points N-rank {data_name}",y_label="N-rank",x_label="Data points",stds=[val[2] for val in y_stds],output_path=output_path)

def evaluate_guides_replicates(guide_data_1, guide_data_2, title, label_column, job, output_path, guides_list = None,
                               features_columns = ['target','offtarget_sequence','chrom','chromStart','chromEnd']):
    '''This function will evaluate the concurence between 2 off target data sets. 
    The function will extract the matching guides from both data sets. 
    If guide list is given the function will evaluate only the guides in the list.
    If job is binary the function will calculate the AUROC, AUPRC and N-rank given the data sets else it will calculate the R2 score and pearson correlation.
    It will consider the first data set as the model and the second data set as the lables and vs versa.
    It will return the metrics for both cases, and plot them togther.
    Args:
    1. guide_data_1 - path to the first data set.
    2. guide_data_2 - path to the second data set.
    3. title - tuple where [0] is the title for the first df, [1] for the second df.
    4. features_columns - columns with the features.
    5. label_column - column with the labels.
    6. job - binary classification/ regression.
    7. output_path - path to save the plots.
    8. guides_list - list of guides to evaluate.
    Example:
    gs_hendel = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv"
    gs_change = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivosilico_nobulges_withEpigenetic_indexed.csv"
    evaluate_guides_replicates(gs_hendel, gs_change, ("Hendel","CHANGE-seq"), "Read_count", "binary","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq")
    '''
    assert job in ['binary','regression'], "Job must be binary or regression"
    # Read the data
    guide_data_1 = pd.read_csv(guide_data_1)
    guide_data_2 = pd.read_csv(guide_data_2)
    # Filter the data by the guides
    guide_data_1, guide_data_2 = off_target_data_by_intersecting_guides(guide_data_1, guide_data_2, 'target', guides_list)
    # Keep only rows with labels > 0
    guide_data_1 = guide_data_1[guide_data_1[label_column] > 0]
    guide_data_2 = guide_data_2[guide_data_2[label_column] > 0]
    # Merge the data
    merged_df = merge_ots_replicates_values(guide_data_1, guide_data_2, label_column, features_columns, job)
    # Rank the data 1 model,2 lables
    if job == 'binary':
        tpr,fpr,percision,baseline = convert_label_to_tpr_fpr_percision(merged_df, 'label_df1', 'label_df2')
        tpr_2,fpr_2,percision_2,baseline_2 = convert_label_to_tpr_fpr_percision(merged_df, 'label_df2', 'label_df1')
        # Aucs
        aucs = [auc(fpr,tpr),auc(fpr_2,tpr_2)]
        auprcs = [(auc(tpr,percision),baseline),(auc(tpr_2,percision_2),baseline_2)]
        # Fprs
        fprs = [fpr,fpr_2]
        tprs = [tpr,tpr_2]
        percs = [percision,percision_2]
        # Create title list
        title = [f"{title[0]} vs {title[1]}", f"{title[1]} vs {title[0]}" ]
        plot_roc(fpr_list=fprs,tpr_list=tprs,aurocs=aucs,titles=title,output_path=output_path,general_title="intersect_6_roc")
        plot_pr(recall_list=tprs,precision_list=percs,auprcs=auprcs,titles=title,output_path=output_path,general_title="intersect_6_pr")
    else : # Job is regression
        x_labels,y_labels = merged_df['label_df1'].values,merged_df['label_df2'].values
        x_lables_log = x_labels + 1 # add 1 to avoid log(0)
        y_labels_log = y_labels + 1
        x_lables_log,y_labels_log = np.log(x_lables_log),np.log(y_labels_log)
        r,p = pearson_correlation(x_labels,y_labels)
        r_log,p_log = pearson_correlation(x_lables_log,y_labels_log)
        title = f"{title[0]} vs {title[1]}"
        plot_correlation(x=x_labels,y=y_labels,r_coeff=r,p_value=p,title=title,output_path=output_path)
        plot_correlation(x=x_lables_log,y=y_labels_log,r_coeff=r_log,p_value=p_log,title=f'{title} - Log',output_path=output_path)
        pass
def off_target_data_by_intersecting_guides(ots_data_1, ots_data_2, guide_column, guide_list = None):
    '''This function takes two off target data frames and returns only the rows with guides presented in both data frames.
    If guide list given is returns rows with guides presented in the list.
    Args: 
    1. ots_data_1 - first off target data frame.
    2. ots_data_2 - second off target data frame.
    3. guide_column - column with the guides.
    4. guide_list - list of guides to filter.
    ------------
    Returns: ots_data_1, ots_data_2 only with intersecting guides.'''
    if guide_list is None:
        guide_list = set(ots_data_1[guide_column].values).intersection(set(ots_data_2[guide_column].values))
    # Filter the data frames
    ots_data_1 = ots_data_1[ots_data_1[guide_column].isin(guide_list)]  
    ots_data_2 = ots_data_2[ots_data_2[guide_column].isin(guide_list)]
    return ots_data_1, ots_data_2
def merge_ots_replicates_values(ots_data_1, ots_data_2, label_column, features_columns,job):
    '''This function will merge the two off target data sets by the features columns.
    The merge data set will hold the label value from the first df and the second df.
    If there is no label in one of the data sets the label will be set to 0 for that df.
    Args:
    1. ots_data_1 - first off target data frame.
    2. ots_data_2 - second off target data frame.
    3. label_column - column with the labels.
    4. features_columns - columns with the features.
    ------------
    Returns: merged off target data frame.'''
    # Rename the label columns to differentiate them
    ots_data_1 = ots_data_1.rename(columns={label_column: 'label_df1'})
    ots_data_2 = ots_data_2.rename(columns={label_column: 'label_df2'})
    # Keep only the features columns and label columns
    ots_data_1 = ots_data_1[features_columns + ['label_df1']]
    ots_data_2 = ots_data_2[features_columns + ['label_df2']]
    merged_df = pd.merge(ots_data_1, ots_data_2, on = features_columns, how = 'outer')
    # Check size of merged_df
    inner_df = pd.merge(ots_data_1, ots_data_2, on = features_columns, how = 'inner')
    intersecting_length = len(inner_df)
    if job == 'regression': # return the inner merged df
        if intersecting_length <= 0 :
            raise RuntimeError("No intersecting guides between the data frames cannot do regression")
        return inner_df
    only_df1 = len(ots_data_1) - intersecting_length
    only_df2 = len(ots_data_2) - intersecting_length
    if (only_df1 + only_df2 + intersecting_length) != len(merged_df):
        raise RuntimeError("Error in merging the data frames")
    if (merged_df["label_df2"].isna().sum() != only_df1):
        raise RuntimeError("Error in merging the data frames, missing values in label_df2")
    if (merged_df["label_df1"].isna().sum() != only_df2):
        raise RuntimeError("Error in merging the data frames, missing values in label_df1")
    # Fill missing values in label columns with 0
    merged_df['label_df1'].fillna(0, inplace=True)
    merged_df['label_df2'].fillna(0, inplace=True)
    print(f'df1 features: {len(ots_data_1)}, df2 features: {len(ots_data_2)}, intersecting features: {intersecting_length}\nvalues only in df1: {only_df1}, values only in df2: {only_df2}')
    return merged_df
def convert_label_to_tpr_fpr_percision(merged_df, label_1, label_2):
    '''This function converts the labels in the merged data frame to TPR, FPR, Percisions values and tp baseline.
    Args:
    1. merged_df - merged off target data frame.
    2. label_1 - label column from the first data frame.
    3. label_2 - label column from the second data frame.
    ------------
    Returns: TPR, FPR, Percision values and tp baseline.
    TP baseline - Total positive/ total population.'''
    # Create 2d np array - first row predictions, 2'd row labels
    pred_label_array = np.array([merged_df[label_1].values, merged_df[label_2].values])
    # Get the total amount of actual positives
    total_positives = np.sum(pred_label_array[1] > 0)
    total_negatives = np.sum(pred_label_array[1] == 0)
    perc_baseline = total_positives / (total_positives + total_negatives)
    # Sort the array in descending order by the first row
    sorted_indices = np.argsort(pred_label_array[0])[::-1]
  
    # Initialize variables
    tpr_values = [0]
    fpr_values = [0]
    precision_values = [1]
    # Calculate TPR and FPR at each threshold
    true_positives = false_positives = 0

    for indice in (sorted_indices):
        # Calculate the number of true positives and false positives up to the current threshold
        true_positives += (pred_label_array[1][indice] > 0)
        false_positives += (pred_label_array[1][indice] == 0)
        # Calculate TPR
        tpr = true_positives / total_positives if total_positives > 0 else 0
        tpr_values.append(tpr)
        # Calculate FPR
        fpr = false_positives / total_negatives if total_negatives > 0 else 0
        fpr_values.append(fpr)
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        precision_values.append(precision)
    return tpr_values, fpr_values, precision_values, perc_baseline
   
if __name__ == "__main__":
    #performance_by_data_points("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence",50,"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel/Performance_by_parts","vivo-silico-hendel")
    scores_path = ["/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/11_group/1-2-3-4-5-6-7-8-9-10-11_partition/1-2-3-4-5-6-7-8-9-10-11_partition_50/Scores/ensemble_1.csv",
                   "/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/7_partition/7_partition_50/Scores/ensemble_1.csv"]
    titles = ["Hendel on Hendel","CHANGE-seq on CHANGE-seq"]
    plot_roc_pr_for_ensmble_by_paths(scores_path,titles,"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq","Own_Performance")
    