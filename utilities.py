import numpy as np
import os

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
    if header: # not None
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
The last line the file should contain the actual labels so each file should have the same labels'''
def extract_scores_from_files(paths):
    all_scores = []
    for path in paths:
        # Read the file
        scores = np.genfromtxt(path, delimiter=',')
        # Remove last row from scores - slice the last row
        y_test = scores[-1]
        scores = scores[:-1]
        # Add the scores to the scores array
        all_scores.append(scores)
    # Concate the arrays
    all_scores = np.concatenate(all_scores)
    return all_scores,y_test

'''Given a list of paths for csv files containing ensembel combinatiorical results
And given a list with model amounts in each ensmbel to check return a dictionary
with the results of each combinatorical amount from each ensmbel'''
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
if __name__ == "__main__":
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