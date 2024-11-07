import numpy as np
import pandas as pd
from itertools import combinations
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, mean_squared_error
from utilities import create_paths, find_target_folders,get_X_random_indices, extract_scores_labels_indexes_from_files
from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std,plot_roc, plot_correlation, plot_pr
from ml_statistics import get_ensmbels_stats, get_mean_std_from_ensmbel_results, pearson_correlation, spearman_correlation



### Metrics evaluations: AUC, AUPRC, N-rank, 

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
    x_values = x_values / amount_of_points # normalize x values
    calculated_auc = auc(x_values,tpr_arr)
    #calculated_auc = calculated_auc / amount_of_points # normalizied auc
    return calculated_auc,amount_of_points

def evaluate_model(y_test, y_scores, task = None):
    '''This function evaluate the model given its task.
    Args:
    1. y_test - the actual labels.
    2. y_scores - the predicted scores.
    3. task - the task of the model. (classification, regression)
    ----------
    returns: dict of the evaluation metrics in regression or auroc,auprc in classification.
    '''
    if task.lower() == "classification":
        return evaluate_auroc_auprc(y_test, y_scores)
    elif task.lower() == "regression":
        return evalaute_regression(y_test, y_scores)
    else:
        raise RuntimeError(f"Task {task} is not supported")

def evaluate_auroc_auprc( y_test, y_pos_scores_probs):
    # # Calculate AUROC,AUPRC
    fpr, tpr, tresholds = roc_curve(y_test, y_pos_scores_probs)
    auroc = auc(fpr, tpr)
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pos_scores_probs)
    return (auroc,auprc)

def evalaute_regression(y_test, y_scores):
    '''This function evaluate the regression model by calculating the pearson and spearman correlations, it also reports the MSE.
    The evaluation is between all data points, and between only the positive OTSs with label > 0.

    Args:
    1. y_test - the actual labels.
    2. y_scores - the predicted scores.
    ------------   
    Returns: tuple of dicts of {pearson: (correlation,p.value), spearman: (correlation,p.value), MSE: mse}
    (all_data_points, positive_data_points)
    '''
    all_data_points = {"pearson": pearson_correlation(y_test, y_scores), "spearman": spearman_correlation(y_test, y_scores), "MSE": mean_squared_error(y_test , y_scores)}
    pos_y_test = y_test[y_test > 0] # get only the positive OTSs
    pos_y_scores = y_scores[y_test > 0] # get only the predicted positive OTSs
    positive_data_points = {"pearson": pearson_correlation(pos_y_test, pos_y_scores), "spearman": spearman_correlation(pos_y_test, pos_y_scores), "MSE": mean_squared_error(pos_y_test , pos_y_scores)}
    return all_data_points, positive_data_points

## Saving the evaluations results
def save_model_results(classification_tuple = None, regression_dict = None, table = None, ml_feature_key_tuple = None, n_rank = None, tpn_tuple = None, task = None):
    if classification_tuple or regression_dict is None:
        raise RuntimeError("No results to save")
    if table is None:
        raise RuntimeError("No table to save the results")
    if ml_feature_key_tuple is None:
        raise RuntimeError("No model, features, key_left_out given to save the results")
    ml_type, features_description, file_left_out = ml_feature_key_tuple
    if task.lower() == "classification":
        if n_rank is None:
            n_rank = (0,0) # default n_rank
        if tpn_tuple is None:
            tpn_tuple = (0,0,0,0) # default tpn_tuple
        auroc, auprc = classification_tuple
        table = save_classification_results(auroc,auprc,file_left_out,table,tpn_tuple,n_rank,ml_type,features_description)
        return table
    elif task.lower() == "regression":
        pearson = regression_dict["pearson"]
        spearman = regression_dict["spearman"]
        mse = regression_dict["MSE"]
        ots_type,ots_amount = tpn_tuple
        table = saving_regression_results(pearson, spearman, mse, file_left_out, table, ml_type, features_description, ots_type, ots_amount)
        return table
    else:
        raise RuntimeError(f"Task: {task} is not supported")
                
        
def save_classification_results(auroc,auprc,file_left_out,table,Tpn_tuple,n_rank,ml_type,features_description):
        '''This function writes the results of the classification task.
    includes: ml type, auroc, auprc, unpacks 4 element tuple - tp,tn test, tp,tn train.
    features included for training the model
    what file/gRNA was left out.'''
        if list(table.columns) != ['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out']:
            raise RuntimeError("The table columns are not as expected")
        try:
            new_row_index = len(table)  # Get the index for the new row
            table.loc[new_row_index] = [ml_type, auroc, auprc,*n_rank,*Tpn_tuple, features_description, file_left_out]  # Add data to the new row
        except: # empty data frame
            table.loc[0] = [ml_type , auroc, auprc,*n_rank,*Tpn_tuple , features_description , file_left_out]
        return table

def saving_regression_results(pearson, spearman, mse, file_left_out, table, ml_type, features_description, OTSs_type, OTSs_amount):
    '''This function writes the results of the regression task.
    includes: 'ML_type', 'R_pearson','P.pearson','R_spearman','P.spearman','MSE','OTSs','N', 'Features', 'File_out'.
    Where OTS is positive OTSs or all OTSs.'''
    if list(table.columns) != ['ML_type', 'R_pearson','P.pearson','R_spearman','P.spearman','MSE','OTSs','N', 'Features', 'File_out']:
        raise RuntimeError("The table columns are not as expected")
    try:
        new_row_index = len(table)  # Get the index for the new row
        table.loc[new_row_index] = [ml_type, pearson[0], pearson[1], spearman[0], spearman[1], mse,OTSs_type, OTSs_amount, features_description, file_left_out]  # Add data to the new row
    except: # empty data frame
        table.loc[0] = [ml_type, pearson[0], pearson[1], spearman[0], spearman[1], mse,OTSs_type, OTSs_amount, features_description, file_left_out]
    return table

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
    ### Test on LAZARATO
    scores_path = ["/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/7_partition/7_partition_50/Scores/ensemble_1.csv",
                    "/localdata/alon/ML_results/Hendel/vivo-silico/test_on_changeseq/6_intersect/all_6/Scores/ensemble_1.csv",
                    "/localdata/alon/ML_results/Hendel_Changeseq/vivo-silico/test_on_changeseq/6_intersecting/all_6/Scores/ensemble_1.csv"]
    ### Test on HENDEL
    scores_path = ["/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/test_on_hendel/6_intersect/all_6/Scores/ensemble_1.csv",
    "/localdata/alon/ML_results/Hendel/vivo-silico/Performance-increasing-OTSs-gRNAs/11_group/1-2-3-4-5-6-7-8-9-10-11_partition/1-2-3-4-5-6-7-8-9-10-11_partition_50/Scores/ensemble_1.csv",
    "/localdata/alon/ML_results/Hendel_Changeseq/vivo-silico/test_on_hendel/6_intersecting/all_6/Scores/ensemble_1.csv"]
    titles = ["L","H","H + L"]

    
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
        auroc, auprc = evaluate_auroc_auprc(y_test, y_scores)
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
    '''This function extract results for each subset in the subsets folder AND for each single feature in the subset
    Args:
    1. subsets_path - path to the subsets folder.
    2. feature_dict - dictionary with the results for each singel feature.
    3. n_models_in_ensmbel - number of models in the ensmbel.
    ------------
    Returns: dictionary with the results for each subset and each single feature in the subset.
    '''
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

def bar_plot_ensembels_feature_performance(only_seq_combi_path, epigenetics_path, n_models_in_ensmbel, plots_output_path, data_output_path, title,  add_subsets = None):
    '''This function will plot the performance of the ensembles for the given features compared with only sequence
    Will create 3 diffrenet plots for AUROC,AUPRC,N-rank each with the performance of the ensembles for the given features
    The bar plots will be created and statistical test between only sequence and the other features will be calculated
    p value marks - *,**,*** will be added to the plot
    ------------
    If add subsets is given - subsets - path for subsets folders , the function will add the performance of the subsets
    and each single feature in the subset to the plot
    Args:
    1. only_seq_combi_path - path to the only sequence combinatorical folder.
    2. epigenetics_path - path to the epigenetics folder with all marks/features - each feature is a folder with combinatorical folder.
    3. n_models_in_ensmbel - number of models in the ensmbel.
    4. plots_output_path - path to save the plots.
    5. data_output_path - path to save the data for future tasks.
    6. title - title for the plot.
    7. add_subsets - path to the subsets marks/features folder.
    -------------
    Returns: None
    Saves: 3 bar plots for AUROC,AUPRC,N-rank and the data for the plots. The plots are:
        1. single features.
        2. subset features if given.
        3. combination of all features.
    Example:
    only_seq_path = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Change_seq/CNN/Ensemble/Only_sequence/1_partition/1_partition_50/Combi"
    epigenetic_path = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Change_seq/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary"
    n_models = 50
    plots_output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/epigenetics"
    data_output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/ensembles/change_seq/epigenetics"
    title = "Only-seq vs Epigenetics"
    add_subsets = "/localdata/alon/ML_results/Change-seq/vivo-silico/Change_seq/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/picked_marks"
    bar_plot_ensembels_feature_performance(only_seq_path, epigenetic_path, n_models, output_path, title, add_subsets)
    '''
    feature_dict = eval_ensembles_in_folder(epigenetics_path, n_models_in_ensmbel, True)
    dict_50_only_seq = extract_combinatorical_results(only_seq_combi_path, [n_models_in_ensmbel])
    feature_dict["Only-seq"] = dict_50_only_seq
    feature_stats = get_ensmbels_stats(feature_dict,50)
    feature_scores = get_mean_std_from_ensmbel_results(feature_dict)
    # Plot single features
    plot_all_ensmbels_means_std(feature_scores,f'{title}_singel_features', feature_stats, plots_output_path, n_models_in_ensmbel)
    if add_subsets:
        subsets_dict =  eval_ensembles_in_folder(add_subsets,n_models_in_ensmbel,True)
        subsets_dict['All'] = feature_dict['All']
        subsets_dict['Only-seq'] = feature_dict['Only-seq']
        subsets_stats = get_ensmbels_stats(subsets_dict,n_models_in_ensmbel)
        subsets_scores = get_mean_std_from_ensmbel_results(subsets_dict)
        # Plot only subsets
        plot_all_ensmbels_means_std(subsets_scores,f"{title}_just_subsets",subsets_stats, plots_output_path, n_models_in_ensmbel)
        feature_dict.update(subsets_dict)
        feature_stats.update(subsets_stats)
        feature_scores.update(subsets_scores)
    # Plot all features
    plot_all_ensmbels_means_std(feature_scores,f'{title}_all_features', feature_stats, plots_output_path, n_models_in_ensmbel)
    # Save the data
    save_bar_plot_data(feature_scores, feature_stats, data_output_path, title)
def save_bar_plot_data(scores_dict, stats_dict, output_path, title):
    '''This function save the raw data used to create the bar plot for future tasks such as:
    reproducing the bar plot without running the whole code, statistical tests and more.
    It will create a csv file with the scores (mean,std) for each feature and the statistical tests results.
    Args:
    1. scores_dict - dictionary with the scores for each feature. {key: feature, value: dict{key: n_models,val: np.array([auroc,auprc,n-rank],[stds])}}
    2. stats_dict - dictionary with the statistical tests results. {key: feature, value: np.array([auroc_pval,auprc_pval,n-rank_pval])}
    3. output_path - path to save the data.
    4. title - title for the data.
    --------------
    Returns: None
    '''
    data = []
    # Iterate through each feature and populate the list with rows of data
    for feature, score_data in scores_dict.items():
        n_models, scores_stds = list(score_data.items())[0]
        scores, stds = scores_stds[0], scores_stds[1]
        pvals = stats_dict[feature] if feature in stats_dict else [0,0,0]
        
        # Append a row with the feature name and corresponding data
        data.append([feature, scores[0], stds[0],pvals[0], scores[1], stds[1],pvals[1], scores[2], stds[2],pvals[2]])

    # Create the DataFrame with columns specified
    df = pd.DataFrame(data, columns=['feature', 'auroc', 'auroc_std','auroc_pval', 'auprc', 'auprc_std','auprc_pval', 'n-rank', 'n-rank_std','n-rank_pval'])

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_path,f'{title}_data.csv'), index=False)

### ENSEMBLE AND MODEL EVALUATIONS ###

def performance_by_data_points(base_path, n_models_in_ensmbel,output_path, counts_path =  None, counts_sgRNA = False, counts_OTSs = False):
    '''This function calculate the prediction performance evaluations over diffrenet number of training data points
    and plot the metric in y axis over data points in the x axis.
    Given a base path with results of ensembles tested on different number of data points (sgRNA, or OTSs)
    Where different number of data_points located in different folders the function will:
    Retrive the combi results from each folder - combine them and calculate the 
    Mean and STD AUPRC, AUROC, N-rank for each amount of data point.
    Plot each amount of data points and the mean performance with std.
    Args:
    1. base_path - path to the base folder with the results.
    2. n_models_in_ensmbel - number of models in the ensmbel.
    3. output_path - path to save the plots.
    4. counts_path - path for the counts per group
    5. counts_sgRNA - boolean if the counts are for sgRNA or OTSs. defualt is False.
    6. counts_OTSS - boolean if the counts are for OTSs. default is False.
    
    Example:
    ### Test on LAZARATO
    # performance_by_data_points("/localdata/alon/ML_results/Change-seq/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence",50,
                               "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/vivo-silico/performance",
                               "vivo-silico","/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/Changeseq_sgRNA_counts.csv","Number of sgRNAs")
    
    ### Test on HENDEL
    performance_by_data_points("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-increasing-sgRNAs",50,
                               "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel/Performance_by_parts",
                               "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Hendel_sgRNA_counts.csv",counts_sgRNA=True,counts_OTSs=False)'''
    group_dict = {}
    folders_names = os.listdir(base_path)
    folder_paths = create_paths(base_path)
    for data_group,data_folder in zip(folders_names,folder_paths):
        
      
        group_paths = find_target_folders(data_folder,["Combi"])
        group_paths.sort() # sort by partition number
        group_paths = [os.path.join(path,"Combi") for path in group_paths] # Add Combi folder to each path
        partition_num = len(group_paths)
        # set array for the values - auroc,auprc,n-rank for each partition in the group
        values_arr = np.zeros(shape=(partition_num,3)) 
        for partition,path in enumerate(group_paths):
            values_arr[partition] = extract_combinatorical_results(path,[n_models_in_ensmbel])[n_models_in_ensmbel]
        data_group = data_group.replace("group","") # Remove the group notation
        data_group = data_group.replace("_"," ") # Remove the underscore notation
        data_group = float(data_group) # convert to float
        group_dict[data_group] = values_arr.mean(axis=0),values_arr.std(axis=0)  
    # Sort dict by keys
    group_dict = dict(sorted(group_dict.items()))
    x_vals = [key for key in group_dict.keys()] # defualt x values by group number
    x_label = "Group number" # defualt x label
    title = "increasing by group number" # defualt title
    scaling = False # Defualt scaling is False
    if counts_path: # if counts given
        if counts_sgRNA:
            x_vals = increasing_points(counts_path, True)
            x_label = "Number of sgRNAs"
            title = "increasing-sgRNAs"
        elif counts_OTSs:
            x_vals = increasing_points(counts_path, False)
            x_label = "Number of OTSs"
            title = "increasing-OTSs"
            scaling = True
        else: # count path is given but not spesified for OTS/sgRNA
            print("Count path is given but no bool for sgRNA/OTSs, setting the x axis values to the group numbers")
            

    else: 
        print ("No count path is given, setting the x axis values to the group numbers")
        
    if len(x_vals) != len(group_dict): # if the amount of x values is not equal to the amount of groups then one group used for testing
        x_vals = x_vals[:-1]
    y_vals = [value[0]  for value in group_dict.values()] 
    y_stds = [value[1] for value in group_dict.values()]
    plot_ensemeble_preformance(y_values=[val[0] for val in y_vals],x_values=x_vals,title=f"Performance by data points AUROC {title}",y_label="AUROC",x_label=x_label,stds=[val[0] for val in y_stds],output_path=output_path,if_scaling=scaling)
    plot_ensemeble_preformance(y_values=[val[1] for val in y_vals],x_values=x_vals,title=f"Performance by data points AUPRC {title}",y_label="AUPRC",x_label=x_label,stds=[val[1] for val in y_stds],output_path=output_path,if_scaling=scaling)
    plot_ensemeble_preformance(y_values=[val[2] for val in y_vals],x_values=x_vals,title=f"Performance by data points N-rank {title}",y_label="N-rank",x_label=x_label,stds=[val[2] for val in y_stds],output_path=output_path,if_scaling=scaling)

def increasing_points(path_to_counts, sgRNA = False):
    '''This function gets path to eather sgRNA counts or OTSs counts per group and returns for each group its amount
    If sgRNA so sums the sgRNAs in increasing order. 
    If OTSs so just return the amount of OTSs in each group
    Args:
    1. path_to_counts - path to the file with the counts in it.
    2. sgRNA_OTS_bool - boolean if the counts are for sgRNA or OTSs.
    True - sgRNA, False - OTSs'''
    counts = []
    if sgRNA: # sgRNA
        counts = pd.read_csv(path_to_counts)["sgRNA Count"].values
        counts = np.cumsum(counts)
    else: # OTSs
        counts = pd.read_csv(path_to_counts)["OTSs count"].values
    return counts
### DIFFERENET DATASETS EVALUATIONS###
def evaluate_guides_replicates(guide_data_1, guide_data_2, title, label_column, job, plot_output_path, data_output_path, guides_list = None,
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
    evaluate_guides_replicates(gs_hendel, gs_change, ("Hendel","CHANGE-seq"), "Read_count", "binary","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq","/home/dsi/lubosha/Off-Target-data-proccessing/Data/Merged_studies")

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
        plot_roc(fpr_list=fprs,tpr_list=tprs,aurocs=aucs,titles=title,output_path=plot_output_path,general_title="intersect_6_roc")
        plot_pr(recall_list=tprs,precision_list=percs,auprcs=auprcs,titles=title,output_path=plot_output_path,general_title="intersect_6_pr")
    else : # Job is regression
        x_labels,y_labels = merged_df['label_df1'].values,merged_df['label_df2'].values
        x_lables_log = x_labels + 1 # add 1 to avoid log(0)
        y_labels_log = y_labels + 1
        x_lables_log,y_labels_log = np.log(x_lables_log),np.log(y_labels_log)
        r,p = pearson_correlation(x_labels,y_labels)
        r_log,p_log = pearson_correlation(x_lables_log,y_labels_log)
        complete_title = f"{title[0]} vs {title[1]}"
        plot_correlation(x=x_labels,y=y_labels,x_axis_label=f'{title[0]} - read count', y_axis_label=f'{title[1]} - read count',r_coeff=r,p_value=p,title=complete_title,output_path=plot_output_path)
        plot_correlation(x=x_lables_log,y=y_labels_log,x_axis_label=f'{title[0]} - read count', y_axis_label=f'{title[1]} - read count',r_coeff=r_log,p_value=p_log,title=f'{complete_title} - Log',output_path=plot_output_path)
    # Save the data
    columns = {"label_df1" : title[0], "label_df2" : title[1]}
    merged_df.rename(columns=columns, inplace=True)
    data_output_path = os.path.join(data_output_path, f"{title[0]}_vs_{title[1]}_{job}.csv")
    merged_df.to_csv(data_output_path, index = False)

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

### FEATURE EVALUTION ###
def feature_correlation(data_path, features_columns, label_column, log_feature = False, log_label = False):
    '''This function will calculate the correlation between the features and the label.
    The function will calculate the pearson correlation and the p value for each feature.
    Args:
    1. data_path - path to the data.
    2. features_columns - columns with the features.
    3. label_column - column with the label.
    4. log_feature - boolean if to log the feature.
    5. log_label - boolean if to log the label.
    ------------
    Returns: dictionary with the features as keys and the (pearson correlation ,p value, x_values, y_values).'''
    # Read the data
    data = pd.read_csv(data_path)
    # Calculate the pearson correlation and p value for each feature
    correlation_dict = {}
    label_values = data[label_column].values
    if log_label:
        label_values_log = np.log(label_values + 1)
    for feature in features_columns:
        feature_values = data[feature].values
        r, p = pearson_correlation(feature_values, label_values)
        correlation_dict[(feature,label_column)] = (r, p, feature_values, label_values)
        if log_label:
            r_log, p_log = pearson_correlation(feature_values, label_values_log)
            correlation_dict[(feature,f'Log {label_column.lower()}')] = (r_log, p_log, feature_values, label_values_log)
    return correlation_dict
def plot_feature_correlation(data_path, output_path,  feature_columns, label_column):
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
    correlation_dict = feature_correlation(data_path, feature_columns, label_column, log_label = log)
    # Plot the correlation
    for feature_nd_label, values in correlation_dict.items():
        r, p, x_values, y_values = values
        feature,label = feature_nd_label
        feature = feature.replace("_"," ")
        label = label.replace("_"," ")
        plot_correlation(x=x_values, y=y_values, x_axis_label=feature + " score", y_axis_label=label, r_coeff=r, p_value=p, title=feature + " " + label, output_path=output_path)
### METRICS HELPER FUNCTIONS ###
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
    
    plot_feature_correlation("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism_with_model_scores.csv",
                             "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel/Feature_correlation/MOFF",["MOFF","GMT"],"Read_count")
    #  gs_hendel = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv"
    #  gs_change = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/vivosilico_nobulges_withEpigenetic_indexed.csv"
    #  evaluate_guides_replicates(gs_hendel, gs_change, ("Hendel","Lazzarotto et. Al"), "Read_count", "regression","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq","/home/dsi/lubosha/Off-Target-data-proccessing/Data/Merged_studies")
    # models_paths=create_paths()
    # plot_roc_pr_for_ensmble_by_paths()
    # scores_path = ["/localdata/alon/ML_results/Change-seq/vivo-silico/CNN/Ensemble/Only_sequence/7_partition/7_partition_50/Scores/ensemble_1.csv",
    #                 "/localdata/alon/ML_results/Hendel/vivo-silico/test_on_changeseq/6_intersect/all_6/Scores/ensemble_1.csv",
    #                 "/localdata/alon/ML_results/Hendel_Changeseq/vivo-silico/test_on_changeseq/6_intersecting/all_6/Scores/ensemble_1.csv"]
    # titles = ["L","H","H + L"]

    # plot_roc_pr_for_ensmble_by_paths(scores_path,titles,"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel_vs_Change-seq","Test_on_Lazzarotto")
    