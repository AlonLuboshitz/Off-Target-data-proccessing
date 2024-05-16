import numpy as np
from itertools import combinations
import os
from sklearn.metrics import roc_curve, auc, average_precision_score
from utilities import create_paths
from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std
from ml_statistics import get_ensmbels_stats, get_mean_std_from_ensmbel_results


def get_tpr_by_n_expriments(predicted_vals,y_test,n):
    '''get the true positive rate for up to n expriemnets by calculating:
the first n prediction values, what the % of positive predcition out of the the TP amount.
calculate auc value for 1-n'''
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
def get_200_random_indices(n_models, k):
    # Create a list of indices from 0 to n_models
    indices = list(range(n_models))
    all_indices = []
    # Create 200 random combinations of k indices
    for j in range(200):
        random_indices = np.random.choice(indices, k, replace=False)
        all_indices.append(random_indices)
    return all_indices

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
            indices_combinations = get_200_random_indices(n_models, k)
        indices_combinations = [list(indices) for indices in indices_combinations]
    else :
        indices_combinations = get_200_random_indices(n_models, k)    
    
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
