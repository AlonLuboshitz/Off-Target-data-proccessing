from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER
from Server_constants import HENDEL_DICT,CHANGESEQ_DICT, MERGED_DICT

from constants import FEATURES_COLUMNS
from multiprocessing import Pool

from file_management import File_management
#from run_models import run_models
from utilities import set_epigenetic_features_by_string, split_epigenetic_features_into_groups, create_guides_list
from utilities import write_2d_array_to_csv, create_paths, keep_only_folders, add_row_to_np_array, extract_scores_labels_indexes_from_files
from utilities import get_k_choose_n, find_target_folders, convert_partition_str_to_list, keep_positives_by_ratio, keep_negatives_by_ratio
from evaluation import eval_all_combinatorical_ensmbel, bar_plot_ensembels_feature_performance
from features_engineering import generate_features_and_labels
import os
import argparse
import json
import random
import numpy as np

TASK = "Classification"
FEATURE_TYPE = "Only_seq"
MODEL_TYPE = "CNN"




### I want to remove the dependece on file_manager from the run_models, and maybe feature engeriing. 
# The main should accpect the following:
### MOVE ALL PATHS ADDITION FROM RUN_MODELS TO MAIN


# 1. Path to the data and external features data paths. - The main will init file manager to handle the data paths. It will use the 
# file manager to pass the needed arguments to the feature engineering.
# 2. Features to run the model with.
# 2. Type of task for the model - classification, regression. 
# 3. Type of model - CNN,XGboost etc..
# 
# With respect to data path, external feature paths ###

def argparser():
    parser = argparse.ArgumentParser(description='''Python script to init a model and train it on off-target dataset.
                                     Different models, feature types, cross_validations and tasks can be created.
                                     ''')
    parser.add_argument('--model','-m', type=int, 
                        help='''Model number: 1 - LogReg, 2 - XGBoost,
                          3 - XGBoost with class weights, 4 - CNN, 5 - RNN''',
                         required=True, default=4)
    parser.add_argument('--cross_val','-cv', type=int,
                         help='''Cross validation type: 1 - Leave one out, 
                         2 - K cross validation, 3 - Ensmbel''',
                         required=True, default=1)
    parser.add_argument('--features_method','-fm', type=int,
                         help='''Features method: 1 - Only_sequence, 2 - Epigenetics_by_features, 
                         3 - Base_pair_epigenetics_in_Sequence, 4 - Spatial_epigenetics''', 
                        required=True, default = 1)
    parser.add_argument('--features_columns','fc',type=list,
                         help='Features columns - list of string of the features columns in the data', required=False)
    parser.add_argument('--epigenetic_window_size','-ew', type=int, 
                        help='Epigenetic window size - 100,200,500,2000', required=False)
    parser.add_argument('--epigenetic_bigwig','-eb', type=str,
                         help='Path for epigenetic folder with bigwig files for each mark.', required=False)
    parser.add_argument('--task','-t', type=str, help='Task: classification/regression', required=True, default='classification')
    parser.add_argument('--over_sampling','-os', type=str, help='Over sampling: y/n', required=False)
    parser.add_argument('--seed','-s', type=int, help='Seed for reproducibility', required=False)
    parser.add_argument('--data_columns_config','-dcc', type=str, 
                        help='''Path to a json config file with the next columns:
                        target_column, offtarget_column, chrom_column, start_column, end_column, binary_label_column, regression_label_column''',
                         required=True)
    args = parser.parse_args()
    return args

def validate_args(args):
    if args.features_method == 2 and args.features_columns is None:
        raise ValueError("Features columns must be given for epigenetic features")
    if args.features_method == 3 and args.epigenetic_bigwig is None:
        raise ValueError("Epigenetic bigwig folder must be given for base pair epigenetics in sequence")
    if args.features_method == 4 and (args.epigenetic_bigwig is None or args.epigenetic_window_size is None):
        raise ValueError("Epigenetic bigwig folder and epigenetic window size must be given for spatial epigenetics")
    if args.dcc 
def parse_data_columns(json_columns):
    '''This function parse the json columns.'''
    with open(json_columns, 'r') as file:
        config = json.load(file)
        # Access constants from the dictionary
        TARGET_COLUMN = config["TARGET_COLUMN"]
        OFFTARGET_COLUMN = config["OFFTARGET_COLUMN"]
        CHROM_COLUMN = config["CHROM_COLUMN"]
        START_COLUMN = config["START_COLUMN"]
        END_COLUMN = config["END_COLUMN"]
        BINARY_LABEL_COLUMN = config["BINARY_LABEL_COLUMN"]
        REGRESSION_LABEL_COLUMN = config["REGRESSION_LABEL_COLUMN"]
    print(f"Columns: {TARGET_COLUMN}, {OFFTARGET_COLUMN}, {CHROM_COLUMN}, {START_COLUMN}, {END_COLUMN}, {BINARY_LABEL_COLUMN}, {REGRESSION_LABEL_COLUMN}")
def parse_constants_dict(constants_dict):
    global VIVO_SILICO,VIVO_VITRO, ML_RESULTS_PATH, MODELS_PATH, TEST_GUIDES, SILICO_VITRO,TRAIN_GUIDES, DATA_NAME
    VIVO_SILICO = constants_dict["Vivo-silico"]
    VIVO_VITRO = constants_dict["Vivo-vitro"]
    ML_RESULTS_PATH = constants_dict["ML_results"]
    MODELS_PATH = constants_dict["Model_path"]
    TEST_GUIDES = constants_dict["Test_guides"]
    TRAIN_GUIDES = constants_dict["Train_guides"]
    DATA_NAME = constants_dict["Data_name"]
    

## INIT FILE MANAGER
def init_file_management():
    file_manager = File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER, VIVO_SILICO ,VIVO_VITRO)
    set_file_manager_files(file_manager)
    return file_manager
def set_file_manager_files(file_manager):
    file_manager.set_ml_results_path(ML_RESULTS_PATH)
    file_manager.set_models_path(MODELS_PATH)
    file_manager.set_ensmbel_guides_path(TEST_GUIDES)
    set_silico(file_manager)

def set_silico(file_manager):
    file_manager.set_silico_vitro_bools(True, False)
def set_vitro(file_manager):
    file_manager.set_silico_vitro_bools(False, True)
    
## INIT MODEL RUNNER
def init_run_models(file_manager):
    from run_models import run_models
    model_runner = run_models(file_manager)
    return model_runner

def set_up_model_runner():
    file_manager = init_file_management()
    model_runner = init_run_models(file_manager)
    return model_runner, file_manager
def init_cnn(runner):
    runner.set_model(4)
def init_ensmbel(runner):
    runner.set_cross_validation(3)
def init_only_seq(runner):
    runner.set_features_method(1)
def init_epigenetics(runner):
    runner.set_features_method(2)


## SET ENSMBEL PREFERENCES
def set_ensmbel_preferences(file_manager, n_models = None, n_ensmbels = None, partition_num = None):
    
    # Pick partition
    if partition_num is None:
        partition_num = input("Enter partition number: ")
        partition_num = list(partition_num)
    file_manager.set_partition(partition_num)
    # Get guides
    guides_path = file_manager.get_guides_partition()
    guides = []
    for guide_path in guides_path:
        guides += create_guides_list(guide_path, 0) 
    

    # Set n_models in each ensmbel:
    if not n_models: 
        n_models = int(input("Enter number of models in each ensmbel: "))   
        assert n_models > 0, "Number of models must be greater than 0"
    if not n_ensmbels:
        n_ensmbels = int(input("Enter number of ensmbels: "))
        assert n_ensmbels > 0, "Number of ensmbels must be greater than 0"
    file_manager.set_n_models(n_models)
    return  guides, n_models, n_ensmbels

## ENSMBEL
# Only sequence
def create_ensmble_only_seq(partition_num = 0, n_models = 50, n_ensmbels = 10, group_dir = None):
    '''The function will create an ensmbel with only sequence features'''
    runner, file_manager = set_up_model_runner()
    init_cnn(runner)
    init_ensmbel(runner)
    init_only_seq(runner)
    runner.setup_runner()
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager,n_models=n_models, n_ensmbels=n_ensmbels, partition_num=partition_num)
    create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner)


## - EPIGENETICS:
    ## 1. Creation
def create_ensembels_by_feature_columns():
    '''# THIS FUNCTION CAN BE TUREND INTO MULTIPROCESSING!
    Given the epigenetic features columns, the function will split them into groups
    Each group reprsents the epigenetic value estimation i.e binary, score, enrichment.
    The function will create ensmbels for each group and for each feature in the group.'''
    features_dict = split_epigenetic_features_into_groups(FEATURES_COLUMNS)
    arg_list = []
    for group, features in features_dict.items():
        for feature in features:
            create_ensembels_with_epigenetic_features(group, [feature]) # singel epigenetic mark
        create_ensembels_with_epigenetic_features(group, features) # all epigenetic marks togther


def create_ensembels_with_epigenetic_features(group, features,n_models=50):
    '''The function accepet a group type and features in that group.
    It sets the ensmbel prefrences -  i.e. 
    1. The file manager paths.
    2. The runner model algorithem and method type - features columns.
    It add epigenetics/the partition number/ the amount of models to the path
    And call create_n_ensembles to create the ensmbels in that path.'''
    runner, file_manager = set_up_model_runner()
    init_cnn(runner)
    init_ensmbel(runner)
    init_epigenetics(runner)
    runner.setup_runner()
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager=file_manager,n_models=n_models, n_ensmbels=10, partition_num=0)

    if len(features) > 1:
        features_str = "All"
        
    else:
        features_str = "".join(features)    
    temp_suffix = os.path.join(group,features_str) # set path to epigenetic data type - binary, by score, by enrichment.
    file_manager.add_type_to_models_paths(temp_suffix) # add path to train ensmbel
    runner.set_features_columns(features) # set feature   
    create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner)
 
def create_n_ensembles(n_ensembles, n_models, guides, file_manager, runner):
    '''Given number of ensembls to create and n_models to create in each ensmbel
    it will create n_ensembles of n_models'''
    if n_ensembles == 1: # No need for multiprocessing
        output_path = file_manager.create_ensemble_train_folder(1)
        runner.create_ensemble(n_models, output_path, guides, 10)
        return
    # Generate argument list for each ensemble
    ensemble_args_list = [(n_models, file_manager.create_ensemble_train_folder(i), guides,(i*10)) for i in range(1, n_ensembles+1)]
    # Create_ensmbel accpets - n_models, output_path, guides, additional_seed for reproducibility
    # Create a pool of processes
    cpu_count = os.cpu_count()
    num_proceses = min(cpu_count, n_ensembles)
    with Pool(processes=num_proceses) as pool:
        pool.starmap(runner.create_ensemble, ensemble_args_list)
    # for i in range(1,n_ensembles+1):
    #     output_path = file_manager.create_ensemble_train_folder(i)
    #     runner.create_ensemble(n_modles, output_path, guides)


## 2. ENSMBEL SCORES/Predictions

def test_ensembles_via_epi_features_in_folder(models_folder, different_test_folder_path = None):
    '''Given a folder with diffrenet features, each feature is a folder with n ensembls
    Send each folder (feature) to a diffrenet process to create the scores for each ensmbel in the folder
    If a different test folder is given, the function will set this path for the file manager
    The results will be set in this folder.'''
    for path in os.listdir(models_folder):
        test_ensemble_via_epi_feature(os.path.join(models_folder, path), different_test_folder_path)
def test_ensemble_via_epi_feature(model_path, different_test_folder_path):
    '''Given a path to a folder(epigentic feature) with n ensembls the function will create:
    1. file manager
    2. runner models
    init both with cnn,ensmbel, epigenetics
    Will set the file manager and suffix by the ensmbel path and create the score and combi folders
    Will set the runner features column by the suffix
    Call test_ensmbel_scores to test the ensmbel and save the scores in the score folder'''
    runner, file_manager = set_up_model_runner()
    if different_test_folder_path: # Not None
        file_manager.set_ml_results_path(different_test_folder_path)
    init_cnn(runner)
    init_ensmbel(runner)
    init_epigenetics(runner)
    runner.setup_runner()
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager=file_manager,n_models=50, n_ensmbels=10, partition_num=0)
    group_folder = model_path.split("/")[-2]
    epi_mark = model_path.split("/")[-1]
    features_columns_by_epi_mark = set_epigenetic_features_by_string(FEATURES_COLUMNS, epi_mark,"_")
    group_epi_path = os.path.join(group_folder,epi_mark)
    file_manager.add_type_to_models_paths(group_epi_path)
    runner.set_features_columns(features_columns_by_epi_mark)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder() # Create score and combi folders
    ensmbels_paths = create_paths(model_path)  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, guides, score_path) for ensmbel in ensmbels_paths]
    with Pool(processes=10) as pool:
        pool.starmap(test_enmsbel_scores, args)
   

def test_ensemble_via_onlyseq_feature(n_models=50,partition_num = 0,n_ensembels = 10,different_test_folder_path = None, different_test_path = None, group_dir = None, if_multi_process = False):
    runner, file_manager = set_up_model_runner()
    if different_test_folder_path: # Not None
        file_manager.set_ml_results_path(different_test_folder_path)
    if different_test_path:
        file_manager.set_seperate_test_data(different_test_path[0],different_test_path[1])    
    init_cnn(runner)
    init_ensmbel(runner)
    init_only_seq(runner)
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    runner.setup_runner()
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager, n_models=n_models, n_ensmbels=n_ensembels, partition_num=partition_num)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, guides, score_path) for ensmbel in ensmbels_paths]
    if if_multi_process:
        with Pool(processes=10) as pool:
            pool.starmap(test_enmsbel_scores, args)
    else:
        for ensmbel in ensmbels_paths:
            test_enmsbel_scores(runner, ensmbel, guides, score_path)

def test_on_other_data(model_path, test_folder_path, test_guide_path, other_data, silico):
    '''This function test one model performance on other data. For example training the model on
    one data set and testing it on another data set.
    The function creates the file manager and runner.
    Args:
    1. model_path: str - the path to the models folder
    2. test_folder_path: str - the path to save the prediction of the model
    3. test_guide_path: str - the path to the test guides
    4. other_data: list - [paths] paths to the other data [0] silico, [1] vitro
    5. silico - bool, indicating if the data is silico or vitro
    ----------
    example: test_on_other_data(model_path="/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive"
                       ,test_folder_path="/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives",
                       test_guide_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides/test_guides/tested_guides_12_partition.txt",
                       other_data=["/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",None],silico=True)
                       
    model_path="/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive"
    test_folder_path="/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives"
    test_guide_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides/test_guides/tested_guides_12_partition.txt"
    other_data=["/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",None]
    silico=True
    positive_parts = os.listdir(model_path)
    model_paths = [os.path.join(model_path,part) for part in positive_parts]
    test_folder_paths = [os.path.join(test_folder_path,part) for part in positive_parts]
    for test_folder_path in test_folder_paths:
        if not os.path.exists(test_folder_path):
            os.makedirs(test_folder_path)
    args = [(model_path,test_folder_path,test_guide_path,other_data,silico) for model_path,test_folder_path in zip(model_paths,test_folder_paths)]
    with Pool(processes=10) as pool:
        pool.starmap(test_on_other_data,args)'''
    file_manager = File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER, other_data[0] ,other_data[1])
    file_manager.set_ml_results_path(test_folder_path)
    file_manager.set_models_path(model_path)
    try:
        set_silico(file_manager) if silico else set_vitro(file_manager)
    except Exception as e:
        print(e)
    runner = init_run_models(file_manager)
    runner.set_model(4,False)
    runner.set_cross_validation(3,False)
    runner.set_features_method(1,False)
    runner.setup_runner()
    guides = create_guides_list(test_guide_path, 0)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    #ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    #ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    ensmbels_paths = file_manager.get_model_path()
    #for ensmbel in ensmbels_paths:
            #test_enmsbel_scores(runner, ensmbel, guides, score_path)
    test_enmsbel_scores(runner,ensmbels_paths,guides,score_path)
    

def test_enmsbel_scores(runner, ensmbel_path, test_guides, score_path):
    '''Given a path to an ensmbel, a list of test guides and a score path
    the function will test the ensmbel on the test guides and save the scores in the score path.
    Each scores will be added with the acctual label and the index of the data point.'''
    
    print(f"Testing ensmbel {ensmbel_path}")
    models_path_list = create_paths(ensmbel_path)
    models_path_list.sort(key=lambda x: int(x.split(".")[-2].split("_")[-1]))  # Sort model paths by models number
    y_scores, y_test, test_indexes = runner.test_ensmbel(models_path_list, test_guides)
    # Save raw scores in score path
    temp_output_path = os.path.join(score_path,f'{ensmbel_path.split("/")[-1]}.csv')
    y_scores_with_test = add_row_to_np_array(y_scores, y_test)  # add accual labels to the scores
    y_scores_with_test = add_row_to_np_array(y_scores_with_test, test_indexes) # add the indexes of each data point
    y_scores_with_test = y_scores_with_test[:,y_scores_with_test[-1,:].argsort()] # sort by indexes
    write_2d_array_to_csv(y_scores_with_test,temp_output_path,[])

def process_all_ensembels_scores_in_folder(ensmbel_folder):
    '''Given a folder with subfolders inside - each subfolder is a feature
     the feature score will be combinatorical evaluated and saved in the combi folder for each feature'''
    ensmbel_paths = create_paths(ensmbel_folder)
    ensmbel_paths = keep_only_folders(ensmbel_paths)
    ensmbel_paths = [(path,) for path in ensmbel_paths]
    for path in ensmbel_paths:
        process_single_ensemble_scores(*path)
    # with Pool(processes=2) as pool:
    #     pool.starmap(process_ensmbel_scores, ensmbel_paths)
def process_single_ensemble_scores(scores_path,if_multi_process = False):
    '''This function will process all the ensmbel scores in the given path
Given a score csv file it will extract from the scores diffrenet combinations of the scores and evaluate them 
vs the labels. The results will be saved in the combi path for the same ensmbel.'''
    
    file_manager = init_file_management()
    file_manager.set_ml_results_path(scores_path)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    ensmbel_scores_paths = create_paths(score_path) # Get a list of paths for each ensmbel scores
    # Add the combi path to ensmbel paths for the process function
    ensmbel_scores_paths = [(score_path, combi_path) for score_path in ensmbel_scores_paths]
    if if_multi_process:
        # Number of processes in the pool
        num_cores = os.cpu_count()
        num_processes = min(num_cores, len(ensmbel_scores_paths))
        # Create a multiprocessing pool
        with Pool(processes=num_processes) as pool:
            # Map the function to the list of paths
            pool.starmap(process_score_path, ensmbel_scores_paths)
    else:
        for path in ensmbel_scores_paths:
            process_score_path(*path)
        

def process_score_path(score_path,combi_path):
    '''Given a score path containing csv files with predictions score and label scores
    Extract the scores, labels and indexes from the files and evaluate all the combinatorical results.
    Keep the results in the combi_path given'''
    y_scores, y_test, indexes = extract_scores_labels_indexes_from_files([score_path])
    results = eval_all_combinatorical_ensmbel(y_scores, y_test)
    header = ["Auroc", "Auprc", "N-rank", "Auroc_std", "Auprc_std", "N-rank_std"]
    temp_output_path = os.path.join(combi_path, f'{score_path.split("/")[-1]}')
    write_2d_array_to_csv(results, temp_output_path, header)


def set_reproducibility_data(file_manager, run_models, data_path):
    file_manager.set_model_results_output_path(data_path)
    run_models.set_data_reproducibility(True)
    run_models.set_model_reproducibility(False)
def set_reproducibility_models(file_manager, run_models, model_path):
    file_manager.set_model_results_output_path(model_path)
    run_models.set_model_reproducibility(True)
    run_models.set_data_reproducibility(False)

def run_model_only_seq(run_models, model_name):
    run_models.run(True, 1, model_name)
def run_model_only_epigenetic(run_models, model_name):
    run_models.run(True, 2, model_name)
def run_model_seq_nd_epigenetic(run_models, model_name):
    run_models.run(True, 3, model_name)
def run_model_seperate_epigenetics(run_models, model_name):
    run_models.run(True, 4, model_name)
def run_reproducibility_data(run_models, model_name, file_manager, k_times):
    #set_reproducibility_data(file_manager, run_models, DATA_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)
def run_reproducibility_models(run_models, model_name, file_manager, k_times):
    #set_reproducibility_models(file_manager, run_models, MODEL_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)
def run_reproducibility_model_and_data(run_models, model_name, file_manager, k_times):
    # set_reproducibility_data(file_manager, run_models, DATA_REPRODUCIBILITY)
    # set_reproducibility_models(file_manager, run_models, ALL_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)

def create_performance_by_data(partition_amount = 11,n_models=50,n_ensmbels=1):
    '''This function creates ensmbel with n_model for partition unions from the partition amount:
    If partition = N, it will pick up to 10 combinations of N choose i for i in range(1,N+1)
    Each combination will be sent to a diffrenet process to create the ensmbel.
    Args:
    1. partition_amount: int - the amount of partitions to create ensmbels from
    2. n_models: int - the number of models in each ensmbel
    3. n_ensmbels: int - the number of ensmbels in each partition
    ----------------
    Saves the ensmbels in Model path, each partition will have a diffrenet folder named group_{i}.
    Inside that folder there will be a folder for each combination for that partition choose i.'''
    for i in range(1,partition_amount+1):
        indices = get_k_choose_n(partition_amount,i)
        if len(indices) > 10: # Pick up to 10 random indices
            indices = random.sample(indices,10)
        group_dir = f"{i}_group"
        args = [(list(partition),n_models,n_ensmbels,group_dir) for partition in indices]
        processes = min(os.cpu_count(), len(args))
        
        with Pool(processes=processes) as pool:
            pool.starmap(create_ensmble_only_seq, args)
   

def test_performance_by_data(partition_amount = 11, n_models=50, n_ensmbels=1, Models_folder = None, test_path = None, test_guides = None):
    '''The function will test the performance of the ensmbels created by the create_performance_by_data function.
    If the Models_folder is given, the function will test the ensmbels in that folder.
    If the test_path/test_guides is given, the function will test the models on that test path.
    Args:
        partition_amount: int - the amount of partitions created by the create_performance_by_data function
        n_models: int - the number of models in each ensmbel
        n_ensmbels: int - the number of ensmbels each partition holds
        Models_folder: str - the path to the folder containing the ensmbels
        test_path: str - the path to the test data
        test_guides: str - the path to the test guides
        if_partitions: bool - where there are multiple partitions in each group. If true in each group there will be multiple partitions.
        ----------------
        Saves the results in corresponding folder in ML_results path given to the file manager'''
    if Models_folder is not None:
        # Get all the groups in the folder:
        groups =  os.listdir(Models_folder) ## Partitions
        for group in groups:
            # Get all the partitions in the group
            partitions = [convert_partition_str_to_list(partition) for partition in os.listdir(os.path.join(Models_folder,group))]
            args = [(n_models,partition,n_ensmbels,None,(test_path,test_guides),group) for partition in partitions]
            num_processes = min(os.cpu_count(), len(args))
            with Pool(processes=num_processes) as pool:
                pool.starmap(test_ensemble_via_onlyseq_feature, args)
    else:
        for i in range(1,partition_amount+1):
            indices = get_k_choose_n(partition_amount,i)
            if len(indices) > 10: # Pick up to 10 random indices
                indices = random.sample(indices,10)
            group_dir = f"{i}_group"
            args = [(n_models,partition,n_ensmbels,None,(test_path,test_guides),group_dir) for partition in indices]

def performance_by_increasing_positives(data_path, model_path, training_guides_path):
    x_data_points,y_data_points,guides = generate_features_and_labels(data_path=data_path,manager=None,encoded_length=23*6,bp_presenation=6,if_bp=False,
                                 if_only_seq=True,if_seperate_epi=False,epigenetic_window_size=2000,features_columns=None,
                                 if_data_reproducibility=False)
    x_negative_data_points,y_negative_labels = keep_negatives_by_ratio(x_data_points,y_data_points,1)
    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    x_features_by_ratio = []
    y_labels_by_ratio = []
    outputs_path_by_ratio = []
    for ratio in ratios:
        x_features,y_labels = keep_positives_by_ratio(x_data_points,y_data_points,ratio)
        # turn y_labels into binary
        for guide_labels in y_labels:
            guide_labels[:] = 1
            
        
        # concatenate the negative data points
        merged_x = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(x_features, x_negative_data_points)]
        merged_y = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(y_labels, y_negative_labels)]
        for x,y in zip(merged_x,merged_y):
            if len(x) != len(y):
                raise RuntimeError(f"X: {len(x)} Y: {len(y)}")

        x_features_by_ratio.append(merged_x)
        y_labels_by_ratio.append(merged_y)
        outputs_path_by_ratio.append(os.path.join(model_path,f"_{ratio}"))
   
    # Create multiproccesses args for each ratio
    test_guides = set(create_guides_list("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides/test_guides/tested_guides_12_partition.txt",0))
    train_guides = set(guides) - test_guides
    train_guides = list(train_guides)
    
    args = [(50,path,train_guides,10,x_merged,y_merged,guides) for path,x_merged,y_merged in zip(outputs_path_by_ratio,x_features_by_ratio,y_labels_by_ratio)]
    
    min_processes = min(os.cpu_count(),int(len(args)/1))
    with Pool(processes=min_processes) as pool:
        pool.starmap(positive_ratio_multi_process,args)
def positive_ratio_multi_process(n_ensmbels,path,train_guides,seed,x_features,y_labels,all_guides):
    try:
        file_manager = File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,"/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv" , "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv")
        runner = init_run_models(file_manager)
    except Exception as e:
        print(e)
    runner.set_model(4,False)
    runner.set_cross_validation(3,False)
    runner.set_features_method(1,False)
    runner.setup_runner()
    runner.create_ensemble(n_ensmbels,path,train_guides,seed,x_features,y_labels,all_guides)

def performance_by_data(train, test, evalute):
    '''This function run the analysis to evaluate the performance of models/ensmbels while increasing the data amount.
    Args:
        Each arg is a dict with the following: 
        train: {"if_create": bool, "partition_amount": int, "n_models": int, "n_ensmbels": int}
        test: {"If_test": bool, "partition_amount": int, "n_models": int, "n_ensmbels": int, 
                "Models_folder": str, "test_path": str, "test_guides": str}
        evalute: bool - if True will evaluate the ensmbels'''
    if train["if_create"]:
        create_performance_by_data(True)
def combiscore_by_folder(base_path):
    scores_nd_combi_paths = find_target_folders(base_path, ["Scores", "Combi"])
    
    # turn the list into list of tuples (for multi process)
    scores_nd_combi_paths = [(path,False) for path in scores_nd_combi_paths]
    
    with Pool(processes=10) as pool:
        pool.starmap(process_single_ensemble_scores,scores_nd_combi_paths)
### With creation of ensemble i seed is *100 and not 10! need to change back!
def only_seq_ensemble_pipe():
    create_ensmble_only_seq(partition_num=[0],n_models=50,n_ensmbels=10)
    #test_ensemble_via_onlyseq_feature(n_models=50,different_test_folder_path="/localdata/alon/ML_results/Train_vitro_test_genome")
    #process_single_ensemble_scores("/localdata/alon/ML_results/Train_vitro_test_genome/CNN/Ensemble/Only_sequence/1_partition/1_partition_50")
    pass
def epigeneitc_ensemble_pipe():
    from Server_constants import LOCAL_RESULTS_EPIGENETICS, LOCAL_MODELS_EPIGENETICS
    create_ensembels_by_feature_columns()
    #test_ensembles_via_epi_features_in_folder(LOCAL_MODELS_EPIGENETICS,"/localdata/alon/ML_results/Train_vitro_test_genome")
    #process_all_ensembels_scores_in_folder("/localdata/alon/ML_results/Train_vitro_test_genome/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary")
    pass

if __name__ == "__main__":
    parse_constants_dict(MERGED_DICT)
    #performance_by_increasing_positives("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv","/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive","")
    #epigeneitc_ensemble_pipe()
    #only_seq_ensemble_pipe()
    #create_ensmble_only_seq(partition_num=[1],n_models=50,n_ensmbels=1)
    #create_performance_by_data()
    # test_performance_by_data(Models_folder="/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive",
    #                          test_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",
    #                          test_guides="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides/tested_guides_12_partition.txt")
    #combiscore_by_folder("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives")
    # bar_plot_ensembels_feature_performance(only_seq_combi_path="/localdata/alon/ML_results/Change-seq/Train_vitro_test_genome/CNN/Ensemble/Only_sequence/1_partition/1_partition_50/Combi",
    #                                        epigenetics_path="/localdata/alon/ML_results/Change-seq/Train_vitro_test_genome/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary",
    #                                        n_models_in_ensmbel=50,output_path="/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/train_vitro_test_silico",
    #                                        title="Train_vitro_test_silico")
    
 
    #process_score_path("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives/_1/Scores/_1.csv","/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives/_1/Combi")
   