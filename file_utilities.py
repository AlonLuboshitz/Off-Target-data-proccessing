# File utilities
import os
import re
from shutil import rmtree
## FILES
def remove_dir_recursivly(dir_path):
    try:
        rmtree(dir_path)
        print(f"Directory '{dir_path}' and its contents have been removed.")
    except Exception as e:
        print(f"Error: {e}") 

def create_paths(folder):
    '''Create paths list off all the files/folders in the given folder'''
    paths = []
    for path in os.listdir(folder):
        paths.append(os.path.join(folder,path))
    return paths

'''create folder in spesefic path'''
def create_folder(path, extend = None):
    if not os.path.exists(path):
        os.makedirs(path)
        print("created new folder: ", path)
    if not extend is None:
        path = os.path.join(path,extend)
        if not os.path.exists(path):
            os.makedirs(path)
            print("created new folder: ", path)
        return path

def keep_only_folders(paths_list):
    '''Given list of paths return only folders from the list'''
    return [path for path in paths_list if os.path.isdir(path)]

def validate_path(path):
    '''Validate if path exists or not'''
    return os.path.exists(path)

def find_target_folders(root_dir, target_subdirs):
    '''This function will iterate the root directory and return paths to the target folders
    '''
    target_folders = []
    for current_dir, dirs, files in os.walk(root_dir):
        # Check if both "Scores" and "Combi" are in the current directory
       if all(subdir in dirs for subdir in target_subdirs):
            target_folders.append(current_dir)
    return target_folders

def extract_ensmbel_combi_inner_paths(base_path):
    '''This function will iterate the base path:
    Base path -> partitions -> inner folders (number of ensmbels) - > Combi
    --------
    Returns a list of paths to the Combi folders from each inner folder'''
    path_lists = []
    for partition in os.listdir(base_path): # iterate partition
        partition_path = os.path.join(base_path,partition)
        for n_ensmbels_path in os.listdir(partition_path): # iterate inner folders
            parti_ensmbels_path = os.path.join(partition_path,n_ensmbels_path)
            if os.path.isdir(os.path.join(parti_ensmbels_path,"Combi")): # if Combi folder exists
                path_lists.append(parti_ensmbels_path)
    return path_lists


def get_bed_folder(bed_parent_folder):
    ''' function iterate on bed folder and returns a list of tuples:
    each tuple: [0] - folder name [1] - list of paths for the bed files in that folder.'''  
    # create a list of tuples - each tuple contain - folder name, folder path inside the parent bed file folder.
    subfolders_info = [(entry.name, entry.path) for entry in os.scandir(bed_parent_folder) if entry.is_dir()]
    # Create a new list of tuples with folder names and the information retrieved from the get bed files
    result_list = [(folder_name, get_bed_files(folder_path)) for folder_name, folder_path in subfolders_info]
    return result_list

def get_bed_files(bed_files_folder):
        
    '''function retrives bed files
    args- bed foler
    return list paths.'''
    bed_files = []
    for foldername, subfolders, filenames in os.walk(bed_files_folder):
        for name in filenames:
            # check file type the narrow,broad, bed type. $ for ending
            if re.match(r'.*(\.bed|\.narrowPeak|\.broadPeak)$', name):
                bed_path = os.path.join(foldername, name)
                bed_files.append(bed_path)
    return bed_files


def copy_ensmebles():
    import os
    import shutil

    # Define the source and destination base directories
    source_base = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/7_partition_50/binary"
    dest_base = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/1_ensembels/50_models/binary"

    # Iterate through all folders in the source binary directory
    for folder in os.listdir(source_base):
        source_scores_dir = os.path.join(source_base,f'{folder}/Scores' )
        dest_scores_dir = os.path.join(dest_base, f'{folder}/Scores')
        
        # Check if the source Scores directory exists
        if os.path.exists(source_scores_dir):
            source_file = os.path.join(source_scores_dir, "ensmbel_1.csv")
            # Check if the file exists before copying
            if os.path.exists(source_file):
                # Create the destination Scores directory if it doesn't exist
                os.makedirs(dest_scores_dir, exist_ok=True)
                # Copy the file to the destination directory
                shutil.copy(source_file, dest_scores_dir)
                print(f"Copied {source_file} to {dest_scores_dir}")
            else:
                print(f"File not found: {source_file}")
        else:
            print(f"Scores directory not found in: {os.path.join(source_base, folder)}")
