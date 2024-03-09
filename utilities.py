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
    if len(header) != np_array.shape[1]:
        raise Exception("header must be the same length as the number of columns in the np_array") 
    np.savetxt(file_path, np_array, delimiter=',', fmt='%.5f', header=','.join(header), comments='')

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
