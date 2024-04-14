import numpy as np
import os
from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std
'''This function is used to validate the input from the user.
 It checks if the input is a valid number and if it is within the range of the list of options
 given by keys of the dictionary'''
def validate_dictionary_input(answer, dictionary):
        assert answer in dictionary.keys(), f"Invalid input. Please choose from {dictionary.keys()}"

'''function takes the guides path and return a list from the i line of the file'''
def create_guides_list(guides_path,i_line):
    with open(guides_path, "r") as f:
        for i, line in enumerate(f):
            if i == i_line:
                line = line.replace("\n","")
                line2 = line.split(",")
                guides = [guide.replace(' ','') for guide in line2]
                break
    return guides
'''Function writes 2d array to csv file'''
def write_2d_array_to_csv(np_array, file_path, header):
    if np_array.ndim != 2:
        raise Exception("np_array must be 2d")
    if file_path.split(".")[-1] != "csv":
        raise Exception("file_name must end with csv")
    if header: # not None/empty
        if len(header) != np_array.shape[1]:
            raise Exception("header must be the same length as the number of columns in the np_array") 
   
    np.savetxt(file_path, np_array, delimiter=',', fmt='%.5f', header=','.join(header), comments='')
def add_row_to_np_array(y_scores, y_test):
    # if y_scores.dtype != y_test.dtype:
    #     raise Exception("y_scores and y_test must have the same dtype")
    if y_scores.ndim != 2 or y_test.ndim != 1:
        raise Exception("y_scores must be 2d and y_test must be 1d")
    if y_scores.shape[1] != y_test.shape[0]:
        raise Exception("y_scores must have the same number of columns as y_test number of values")
    return np.vstack((y_scores, y_test))
    
'''Give a list of features combine the features into groups based on their endings - binary, score, enrichment, etc..'''
def split_epigenetic_features_into_groups(features_columns):
    # Create a dictionary to store groups based on endings
    groups = {}
    # Group strings based on their endings
    for feature in features_columns:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups  
    



## FILES
    
'''Create paths list from folder'''
def create_paths(folder):
    paths = []
    for path in os.listdir(folder):
        paths.append(os.path.join(folder,path))
    return paths
'''Given list of paths return only folders from the list'''
def keep_only_folders(paths_list):
    return [path for path in paths_list if os.path.isdir(path)]
'''Validate if path exists or not'''
def validate_path(path):
    return os.path.exists(path)


'''Given a list of paths for csv files containing models predicitions scores
extract the scores and combine them into one np array.
The last line the file should contain the indexes of each data point
The second raw from the end the actual labels '''
def extract_scores_labels_indexes_from_files(paths):
    all_scores = []
    for path in paths:
        # Read the file
        scores_lbl_idx = np.genfromtxt(path, delimiter=',')
        scores = scores_lbl_idx[:-2] # keep only the scores
        all_scores.append(scores) # Add the scores to the scores array
    # get indices and labels
    indexes = scores_lbl_idx[-1]
    y_test = scores_lbl_idx[-2]
    # Concate the arrays
    all_scores = np.concatenate(all_scores)
    return all_scores,y_test,indexes

'''Given a list of paths for csv files containing ensembel combinatiorical results
And given a list with model amounts in each ensmbel to check return a dictionary
with the results of each combinatorical amount from each ensmbel
Dict returned : {n_keys : (n_ensmbels,[auroc,auprc,n-rank]) }'''
def extract_combinatorical_results(ensmbel_combi_path, n_models_in_ensmbel_list):
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

def plot_ensmbels_means_std():
    list_50 = [i for i in range(2,51)]
    list_40 = [i for i in range(2,41) ]
    list_30 = [i for i in range(2,31)]
    list_20 = [i for i in range(2,21)]
    dict_50 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/1_partition_50/Combi",list_50)
    dict_40 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/1_partition_40/Combi",list_40)
    dict_30 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/1_partition_30/Combi",list_30)
    dict_20 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/1_partition_20/Combi",list_20)
    print(f'mean of 50: {np.mean(dict_50[50],axis=0)}\nmean of 40: {np.mean(dict_40[40],axis=0)}\nmean of 30: {np.mean(dict_30[30],axis=0)}\nmean of 20: {np.mean(dict_20[20],axis=0)}')
    print(f'std of 50: {np.std(dict_50[50],axis=0)}\nstd of 40: {np.std(dict_40[40],axis=0)}\nstd of 30: {np.std(dict_30[30],axis=0)}\nstd of 20: {np.std(dict_20[20],axis=0)}')
    print(f"std of 50,40,30,20 from 50\n{np.std(dict_50[50],axis=0)}\n{np.std(dict_50[40],axis=0)}\n{np.std(dict_50[30],axis=0)}\n{np.std(dict_50[20],axis=0)}")
    print(f"std of 40,30,20 40:\n{np.std(dict_40[40],axis=0)}\n{np.std(dict_40[30],axis=0)}\n{np.std(dict_40[20],axis=0)}")
    print(f"std of 30,20 30:\n{np.std(dict_30[30],axis=0)}\n{np.std(dict_30[20],axis=0)}")
    aurocs_stds_50 = [np.std(dict_50[n],axis=0)[0] for n in list_50]
    auprcs_stds_50 = [np.std(dict_50[n],axis=0)[1] for n in list_50]
    n_ranks_stds_50 = [np.std(dict_50[n],axis=0)[2] for n in list_50]
    plot_ensemeble_preformance(auprcs_stds_50,list_50,"auroc std by n models","Auroc STD","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    plot_ensemeble_preformance(auprcs_stds_50,list_50,"auprc std by n models","Auprc STD","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    plot_ensemeble_preformance(n_ranks_stds_50,list_50,"n-rank std by n models","N-rank STD","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    auroc_mixed_stds = [np.std(dict_50[50],axis=0)[0],np.std(dict_40[40],axis=0)[0],np.std(dict_30[30],axis=0)[0],np.std(dict_20[20],axis=0)[0]]
    auprc_mixed_stds = [np.std(dict_50[50],axis=0)[1],np.std(dict_40[40],axis=0)[1],np.std(dict_30[30],axis=0)[1],np.std(dict_20[20],axis=0)[1]]
    plot_ensemeble_preformance(auroc_mixed_stds,[50,40,30,20],"auroc std by n models - diff ensmbels","Auroc STD","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq") 
    plot_ensemeble_preformance(auprc_mixed_stds,[50,40,30,20],"auprc std by n models - diff ensmbels","Auprc STD","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")

if __name__ == "__main__":
    #list_50 = [i for i in range(2,51)]
    list_50 = [50]
    dict_50_only_seq = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Only_seq/1_partition_50/Combi",list_50)
    dict_50_atac = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Atacseq/Combi",list_50)
    dict_50_h3k4me3 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/H3K4me3/Combi",list_50)
    dict_50_both = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Both/Combi",list_50)
    aurocs_results = [np.mean(dict_50_only_seq[50],axis=0)[0],np.mean(dict_50_atac[50],axis=0)[0],np.mean(dict_50_h3k4me3[50],axis=0)[0],np.mean(dict_50_both[50],axis=0)[0]]
    aurocs_stds = [np.std(dict_50_only_seq[50],axis=0)[0],np.std(dict_50_atac[50],axis=0)[0],np.std(dict_50_h3k4me3[50],axis=0)[0],np.std(dict_50_both[50],axis=0)[0]]
    plot_ensemble_performance_mean_std(aurocs_results,aurocs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auroc by ensmbels - seq,epi","Auroc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    auprcs_results = [np.mean(dict_50_only_seq[50],axis=0)[1],np.mean(dict_50_atac[50],axis=0)[1],np.mean(dict_50_h3k4me3[50],axis=0)[1],np.mean(dict_50_both[50],axis=0)[1]]
    auprcs_stds = [np.std(dict_50_only_seq[50],axis=0)[1],np.std(dict_50_atac[50],axis=0)[1],np.std(dict_50_h3k4me3[50],axis=0)[1],np.std(dict_50_both[50],axis=0)[1]]
    plot_ensemble_performance_mean_std(auprcs_results,auprcs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auprc by ensmbels - seq,epi","Auprc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    n_ranks_results = [np.mean(dict_50_only_seq[50],axis=0)[2],np.mean(dict_50_atac[50],axis=0)[2],np.mean(dict_50_h3k4me3[50],axis=0)[2],np.mean(dict_50_both[50],axis=0)[2]]
    n_ranks_stds = [np.std(dict_50_only_seq[50],axis=0)[2],np.std(dict_50_atac[50],axis=0)[2],np.std(dict_50_h3k4me3[50],axis=0)[2],np.std(dict_50_both[50],axis=0)[2]]
    plot_ensemble_performance_mean_std(n_ranks_results,n_ranks_stds,["Only_seq","Atacseq","H3K4me3","Both"],"N-rank by ensmbels - seq,epi","N-rank","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")