import numpy as np
import os
from plotting import plot_ensemeble_preformance,plot_ensemble_performance_mean_std
from ml_statistics import get_ensmbels_stats

def validate_dictionary_input(answer, dictionary):
    '''This function is used to validate the input from the user.
 It checks if the input is a valid number and if it is within the range of the list of options
 given by keys of the dictionary'''
    if not answer: # no input was given
        answer = input(f"Please enter model: {dictionary}\n") # ask for model
    # input was given
    answer = int(answer)
    if answer not in dictionary.keys(): # check if the input is in the dictionary
        raise Exception(f"Invalid input. Please choose from {dictionary.keys()}")
    else :
        return answer


def create_guides_list(guides_path,i_line):
    '''function path to guides txt file and return a list from the i line of the file
    i_line is the line number to read from the file
    the returned list objects are gRNA strings separted by comma "," '''

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
    
def split_epigenetic_features_into_groups(features_columns):
    '''Give a list of features combine the features into groups based on
      their endings - binary, score, enrichment, etc.
      
      Returns a dictionary with the groups as keys and the features as values.'''

    # Create a dictionary to store groups based on endings
    groups = {}
    # Group strings based on their endings
    for feature in features_columns:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups  

def set_epigenetic_features_by_string(feature_columns, epi_string, split_by):
    '''This function take a string of epigentic features and return a list of the features matching the
the features columns
features_columns - name of the epigenetic features columns
epi_string - string of epigenetic features
split_by - how to split the epi_string
--------
returns a list of the features matching the epi_string'''
    epi_features = epi_string.split(split_by)
    return_list = []
    for feature in epi_features:
        feature = feature.strip().lower()  # Remove whitespace and convert to lowercase
        for column in feature_columns:
            if feature in column.strip().lower():
                return_list.append(column)
    return return_list
    
            

## FILES
    

def create_paths(folder):
    '''Create paths list off all the files/folders in the given folder'''
    paths = []
    for path in os.listdir(folder):
        paths.append(os.path.join(folder,path))
    return paths

def keep_only_folders(paths_list):
    '''Given list of paths return only folders from the list'''
    return [path for path in paths_list if os.path.isdir(path)]
def validate_path(path):
    '''Validate if path exists or not'''
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
def get_mean_std_from_ensmbel_results(ensmbel_results):
    '''Given a dictionary of ensmbel results:
     {key : folder_name, val: dict{key: n_models_in_ensmbel, val: np array of results}
    calculate the mean and std of the results for each ensmbel.
     ------
      returns a dictionary with the mean and std for each ensmbel - [0] -auroc,[1] - auprc,[2] - n-rank'''
    ensmbel_mean_std = {}
    for ensmbel,results in ensmbel_results.items():
        ensmbel_mean_std[ensmbel] = {n_models: (np.mean(results[n_models],axis=0),np.std(results[n_models],axis=0)) for n_models in results.keys()}
    return ensmbel_mean_std
def plot_all_ensmbels_means_std(ensmbel_mean_std_dict,features,stats_dict, n_modles_in_ensmbel = None):
    '''Given a dictionary with the mean and std for each ensmbel - [0] -auroc,[1] - auprc,[2] - n-rank
    plot the results for all ensmbels togther
    stats dict is {Key: group/model, value: pvals of [auroc,auprc,n-rank]}'''
    x_values = [str(ensmbel) for ensmbel in ensmbel_mean_std_dict.keys()]
    aurocs_results = [results[n_modles_in_ensmbel][0][0] for results in ensmbel_mean_std_dict.values()]
    aurocs_stds = [results[n_modles_in_ensmbel][1][0] for results in ensmbel_mean_std_dict.values()]
    roc_pvals = {key: stats_dict[key][0] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(aurocs_results,aurocs_stds,x_values,roc_pvals,f"Auroc by ensmbels - {features}","Auroc",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/epigenetics")
    auprcs_results = [results[n_modles_in_ensmbel][0][1] for results in ensmbel_mean_std_dict.values()]
    # print the model name and auprc results
    for key in ensmbel_mean_std_dict.keys():
        print(key,ensmbel_mean_std_dict[key][n_modles_in_ensmbel][0][1])
    auprcs_stds = [results[n_modles_in_ensmbel][1][1] for results in ensmbel_mean_std_dict.values()]
    prc_pvals = {key: stats_dict[key][1] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(auprcs_results,auprcs_stds,x_values,prc_pvals,f"Auprc by ensmbels - {features}","Auprc",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/epigenetics")
    n_ranks_results = [results[n_modles_in_ensmbel][0][2] for results in ensmbel_mean_std_dict.values()]
    n_ranks_stds = [results[n_modles_in_ensmbel][1][2] for results in ensmbel_mean_std_dict.values()]
    n_rank_pvals = {key: stats_dict[key][2] for key in stats_dict.keys()}
    plot_ensemble_performance_mean_std(n_ranks_results,n_ranks_stds,x_values,n_rank_pvals,f"N-rank by ensmbels - {features}","N-rank",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/epigenetics")

def plot_all_ensmbels_std(ensmbel_mean_std_dict,features):
    x_values = [str(ensmbel) for ensmbel in ensmbel_mean_std_dict.keys()]
    aurocs_stds = [results[ensmbel][1][0] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(aurocs_stds,x_values,f"Auroc stds by ensmbels - {features}","Auroc",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    auprcs_stds = [results[ensmbel][1][1] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(auprcs_stds,x_values,f"Auprc stds by ensmbels - {features}","Auprc",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    n_ranks_stds = [results[ensmbel][1][2] for ensmbel,results in ensmbel_mean_std_dict.items()]
    plot_ensemeble_preformance(n_ranks_stds,x_values,f"N-rank stds by ensmbels - {features}","N-rank",f"/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")

if __name__ == "__main__":
    # #list_50 = [i for i in range(2,51)]
    # list_50 = [50]
    # dict_50_only_seq = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Only_seq/1_partition_50/Combi",list_50)
    # dict_50_atac = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Atacseq/Combi",list_50)
    # dict_50_h3k4me3 = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/H3K4me3/Combi",list_50)
    # dict_50_both = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/Epigenetic_binary/1_partition/1_partition_50/Both/Combi",list_50)
    # aurocs_results = [np.mean(dict_50_only_seq[50],axis=0)[0],np.mean(dict_50_atac[50],axis=0)[0],np.mean(dict_50_h3k4me3[50],axis=0)[0],np.mean(dict_50_both[50],axis=0)[0]]
    # aurocs_stds = [np.std(dict_50_only_seq[50],axis=0)[0],np.std(dict_50_atac[50],axis=0)[0],np.std(dict_50_h3k4me3[50],axis=0)[0],np.std(dict_50_both[50],axis=0)[0]]
    # plot_ensemble_performance_mean_std(aurocs_results,aurocs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auroc by ensmbels - seq,epi","Auroc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    # auprcs_results = [np.mean(dict_50_only_seq[50],axis=0)[1],np.mean(dict_50_atac[50],axis=0)[1],np.mean(dict_50_h3k4me3[50],axis=0)[1],np.mean(dict_50_both[50],axis=0)[1]]
    # auprcs_stds = [np.std(dict_50_only_seq[50],axis=0)[1],np.std(dict_50_atac[50],axis=0)[1],np.std(dict_50_h3k4me3[50],axis=0)[1],np.std(dict_50_both[50],axis=0)[1]]
    # plot_ensemble_performance_mean_std(auprcs_results,auprcs_stds,["Only_seq","Atacseq","H3K4me3","Both"],"Auprc by ensmbels - seq,epi","Auprc","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    # n_ranks_results = [np.mean(dict_50_only_seq[50],axis=0)[2],np.mean(dict_50_atac[50],axis=0)[2],np.mean(dict_50_h3k4me3[50],axis=0)[2],np.mean(dict_50_both[50],axis=0)[2]]
    # n_ranks_stds = [np.std(dict_50_only_seq[50],axis=0)[2],np.std(dict_50_atac[50],axis=0)[2],np.std(dict_50_h3k4me3[50],axis=0)[2],np.std(dict_50_both[50],axis=0)[2]]
    # plot_ensemble_performance_mean_std(n_ranks_results,n_ranks_stds,["Only_seq","Atacseq","H3K4me3","Both"],"N-rank by ensmbels - seq,epi","N-rank","/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq")
    dict = eval_ensembles_in_folder("/localdata/alon/ML_results/Change_seq/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary",50,True)
    dict_50_only_seq = extract_combinatorical_results("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/CNN/Ensemble/Only_sequence/1_partition/1_partition_50/Combi",[50])
    dict["Only_seq"] = dict_50_only_seq
    picked_dict = eval_ensembles_in_folder("/localdata/alon/ML_results/Change_seq/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/picked_marks",50,True)
    picked_dict['Only_seq'] = dict_50_only_seq
    picked_dict['ALL'] = dict["ALL"]
    temp_keys = ''
    for keys in picked_dict.keys():
        if len(temp_keys) < len(keys):
            temp_keys = keys
    temp_keys = temp_keys.split("_")
    for key in temp_keys:
        picked_dict[key] = dict[key]
        
    stats_dict = get_ensmbels_stats(dict,50)
    y = get_mean_std_from_ensmbel_results(picked_dict)
    z = get_mean_std_from_ensmbel_results(dict)
    plot_all_ensmbels_means_std(z,"Only_seq-binary", stats_dict, 50)
    # list_dcits_only_seq = {i : None for i in range(10,81,10)} # 10,20,30,40,50,60,70,80
    # for partition in list_dcits_only_seq.keys():
    #     temp_path = f"/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/CNN/Ensemble/Only_sequence/1_partition/1_partition_{partition}/Combi"
    #     list_dcits_only_seq[partition] = extract_combinatorical_results(temp_path,[partition])
    # list_dcits_only_seq = get_mean_std_from_ensmbel_results(list_dcits_only_seq)
    # plot_all_ensmbels_std(list_dcits_only_seq,"Only_seq")
    
    
    