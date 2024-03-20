import numpy as np
from itertools import combinations

from sklearn.metrics import roc_curve, auc, average_precision_score



'''get the true positive rate for up to n expriemnets by calculating:
the first n prediction values, what the % of positive predcition out of the the TP amount.
calculate auc value for 1-n'''
def get_tpr_by_n_expriments(predicted_vals,y_test,n):
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
'''This function aasses all the possible combinatorical options for k given.
N choose K options.
For exmaple: if there are 10 models and n = 3, the function will average 
the results of all the possible combinations of 3 models out of 10.
The function will return the average of the aurpc,auroc and std over all the combinations.'''
def evaluate_n_combinatorical_models(n_models, n_y_scores, y_test, k):
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
'''Evaluate all the possible combinatorical options for an ensmbel
y_scores is 2d array (N_models, scores), y_test is the accautal labels'''
def eval_all_combinatorical_ensmbel(y_scores, y_test, header = ["Auroc","Auprc","N-rank","Auroc_std","Auprc_std","N-rank_std"]):
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

