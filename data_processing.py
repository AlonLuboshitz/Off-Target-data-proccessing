'''
this is script is to run data processing on:
1. guideseq identified folder wtih identified output files.
2. labeling and merging identified files with positive label py script.
3. creating negative labeling for casoffinder output data.
4. decreasing the matching positions in the positive offtargets eg. guide seq from the negatives
5. doing 3+4 using negative_positive py script.
6. adding chrom labeling using chromatin_labeling py scrpit.
NOTE: this script recives a path to an output foler containing casoffinder outputs 
correspoding to the guideseq sequence.
in order to create this output folder use the positive_label (make_input_txt) to give as input to
casoffinder_runner.py which would output the spesified folder!
'''
import subprocess as sub
import os
import sys
from shutil import rmtree


'''function iterate on folder with guideseq folders with identified folders in them.
return list of paths to those identified folders.'''
def iteratefolder(guideseq_folder):
    identified_folders = []
    # using walk on tree folders to get identified folders.
    for foldername, subfolders, filenames in os.walk(guideseq_folder):
        if 'identified' in subfolders:
            identified_folder_path = os.path.join(foldername, 'identified')
            identified_folders.append(identified_folder_path)
    return identified_folders
'''gets current script dir and runs the positive_label on the lists of paths.'''
def run_positive_labeling(path_to_script,path_list):
    # get number of duplicates and erasing data info. (args for positive)
    duplicates = input("please enter number of duplicates per exp (int): ")
    erase = input("please enter True or False for erasing the identified_label folder created: ")
    assert int(duplicates) > 0,"number of duplicates smaller then 1 please start agian!"
    # join path to script
    positive_script_path = "python " + os.path.join(path_to_script,f'positive_label.py')
    # iterate on files in the guideseq path to label them
    for path in path_list:
        temp_arg = positive_script_path + " " + path + " " + str(duplicates) + " " + erase
        print("runnning positive labeling on: {}".format(path))
        labeling = sub.run(temp_arg,capture_output=True,text=True,shell=True,check=True)
        print(labeling.stdout)
''' run the negeative positive function to join inactive and on active off target sites.'''
def run_negative_positive(labeled_paths,path_to_script,output_folder):
    # get path to script.
    negative_positive_script = "python " + os.path.join(path_to_script,f'negative_positive.py')
    # iterate on guideseq labeled files.
    for path in labeled_paths:
        # create args for function.
        temp_arg = negative_positive_script + " " + output_folder + " " + path
        print("runnning negative_positive on: {}".format(path))
        combined = sub.run(temp_arg,capture_output=True,text=True,shell=True,check=True)
        print(combined.stdout)
    # delete casoffinder_output_csv folder and files
    path_to_csv = os.path.join(os.path.dirname(output_folder),f'casoffinder_outputs_csvs')
    try:
        rmtree(path_to_csv)
        print(f"Directory '{path_to_csv}' and its contents have been removed.")
    except Exception as e:
        print(f"Error: {e}")
'''function gets list of paths to guideseq folder and returns a list of paths
to the _lebeled_merge_folders'''
def path_to_merged_files(identified_paths):
    labeled_merged_paths = []
    for path in identified_paths:
        temp_path = os.path.join(os.path.dirname(path),f'_label_sub_only_merged')
        labeled_merged_paths.append(temp_path)
    return labeled_merged_paths  

'''function args: 1 - bed folder path
2 - path for guideseq folder, inside combined files needs chrom labeling
3 - path for current script dir
function calls chromatin_labeling.py script with:
combined file path, bed file path, and the type of chrominfo
'''
def run_chrom_labeling(bed_parent_folder_path,guideseq_folder,script_dir_path):
    # create script path commnad:
    chrom_labeling_script = "python " + os.path.join(script_dir_path,f'chromatin_labeling.py')
    temp_arg = chrom_labeling_script + " " + bed_parent_folder_path + " " + guideseq_folder
    try: 
        chrom_labeling = sub.run(temp_arg,capture_output=True,text=True,shell=True,check=True)
        print(chrom_labeling.stdout)
    except sub.CalledProcessError as e:
        print(e)
 
'''
argv 1 - path to guideseq folders.
argv 2 - path to output (casoffinder)
argv 3 - path to parent bed folder'''
if __name__ == '__main__':
    # iterate on path to identified files and send to positive_label func.
    identified_paths = iteratefolder(sys.argv[1])
    # Get the directory where the current script is located
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    # Positive runner
    run_positive_labeling(current_script_directory,identified_paths)
    # get paths for merged_labeled
    labeled_paths = path_to_merged_files(identified_paths)
    # negative\positive runner
    run_negative_positive(labeled_paths,current_script_directory,sys.argv[2])
    # run chrom_labeling on the guideseq combined output foler
    #run_chrom_labeling(sys.argv[3],sys.argv[1],current_script_directory)


    

    
