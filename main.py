
from multiprocessing import Pool

from file_management import File_management
from file_utilities import create_paths, keep_only_folders
from evaluation import evaluation
from utilities import   print_dict_values, get_memory_usage
from utilities import write_2d_array_to_csv,  add_row_to_np_array, validate_non_negative_int
from utilities import get_k_choose_n,  convert_partition_str_to_list, keep_positives_by_ratio, keep_negatives_by_ratio
from data_constraints_utilities import with_bulges
from k_groups_utilities import get_k_groups_ensemble_args, create_guides_list,get_k_groups_guides, partition_data_for_histograms
from train_and_test_utilities import split_by_guides
from features_engineering import generate_features_and_labels, keep_indexes_per_guide
from features_and_model_utilities import get_features_columns_args_ensembles, parse_feature_column_dict,split_epigenetic_features_into_groups, get_feature_column_suffix
from parsing import features_method_dict, cross_val_dict, model_dict,encoding_dict, early_stoping_dict
from parsing import class_weights_dict,off_target_constrians_dict, main_argparser, parse_args, validate_main_args
from time_loging import log_time, save_log_time, set_time_log
from ensemble_utilities import get_scores_combi_paths_for_ensemble
import os
import json
import random
import numpy as np
import sys
import time
import atexit

global ARGS, PHATS, COLUMNS, TRAIN, TEST, MULTI_PROCESS





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

def set_args(argv):
    parser = main_argparser()
    args = parse_args(argv, parser)
    global ARGS, PHATS, COLUMNS
    ARGS, PHATS, COLUMNS = validate_main_args(args)
    print_dict_values(PHATS)
    print_dict_values(COLUMNS)
    time.sleep(1)

def set_multi_process(gpu_availability):
    '''
    This function set multi process to True if the gpu is NOT available.
    Other wise use the GPU without multiprocessing.
    '''
    global MULTI_PROCESS
    if not gpu_availability:
        MULTI_PROCESS = True
    else:
        MULTI_PROCESS = False

        

def set_cross_val_args(file_manager, train = False, test = False, cross_val_params = None, cross_val = None ):
    '''This function sets the cross validation arguments for the file manager.
    If no cross vaildation method given the fucntion will use the argument given to main.py
    For leave_one_out will set the parameters for file manager needed for leave one out
    For k_groups will set the parameters for file manager needed for k_groups
    For ensemble will set the parameters for file manager needed for ensemble amd return the t_guides, n_models, n_ensmbels'''
    if not cross_val:
        cross_val = ARGS.cross_val
    if not cross_val_params:
        cross_val_params = (ARGS.n_models, ARGS.n_ensmbels, ARGS.partition)
    if cross_val == 1: # leave_one_out
        pass
    elif cross_val == 2: # K_groups
        if ARGS.test_on_other_data: ### Testing on other data -> testing on all the guides in that data
            return None, None, None
        guide_path = file_manager.get_guides_partition_path(train,test)
        if len(ARGS.partition) == 1:
            ARGS.partition = np.arange(1,ARGS.partition[0]+1)
        t_guides = get_k_groups_guides(guide_path,ARGS.partition,train,test)
        return t_guides, None, None
    elif cross_val == 3: # Ensemble
        n_models, n_ensmbels, partition_num = parse_ensemble_params(*cross_val_params)
        file_manager.set_partition(partition_num, train, test)
        file_manager.set_n_ensembels(n_ensmbels)
        file_manager.set_n_models(n_models)
        t_guides = file_manager.get_guides_partition()
        return t_guides, n_models, n_ensmbels

def parse_ensemble_params(n_models = None, n_ensmbels = None, partition_num = None):
    '''This function parse given ensemble parameters. 
    If no parameters are given the function will set the parameters given in the arguments to main.py.
    -----------
    Returns: n_models, n_ensmbels, partition_num'''
    if not n_models:
        n_models = ARGS.n_models
    if not n_ensmbels:
        n_ensmbels = ARGS.n_ensmbels
    if not partition_num:
        partition_num = ARGS.partition
    n_models = validate_non_negative_int(n_models)
    n_ensmbels = validate_non_negative_int(n_ensmbels)
    return n_models, n_ensmbels, partition_num



def set_job(args):
    train = test = process = evaluation = False
    if args.job.lower() == 'train':
        train = True
    elif args.job.lower() == 'test':
        test = True
    elif args.job.lower() == 'process':
        process = True
    elif args.job.lower() == 'evaluation':
        evaluation = True
    else: raise ValueError("Job must be either Train/Test/Evaluation/Process")
    return train, test, process, evaluation


 

### INITIARS ### 
def init_file_management(params=None, phats = None):
    '''This function creates a file management object with the given parameters.'''
    global PHATS
    if not phats: # None
        phats = PHATS
    file_manager = File_management(models_path=phats["Model_path"], ml_results_path=phats["ML_results_path"], 
                                   guides_path=phats["Guides_path"], vivo_silico_path=phats["Vivo-silico"], 
                                   vivo_vitro_path=phats["Vivo-vitro"], epigenetics_bed=phats["Epigenetic_folder"], 
                                   epigenetic_bigwig=phats["Big_wig_folder"], 
                                   partition_information_path=phats["Partition_information"], plots_path=phats["Plots_path"])
    if not params: # None
        ml_name, cross_val, feature_type,epochs_batch,early_stop = model_dict()[ARGS.model], cross_val_dict()[ARGS.cross_val], features_method_dict()[ARGS.features_method], ARGS.deep_params, ARGS.early_stoping[0]
    else:
        ml_name, cross_val, feature_type,epochs_batch,early_stop = params
    early_stop = early_stoping_dict()[early_stop]
    cw,encoding_type,ots_constraints = class_weights_dict()[ARGS.class_weights], encoding_dict()[ARGS.encoding_type], off_target_constrians_dict()[ARGS.off_target_constriants]
    file_manager.set_model_parameters(data_type=ARGS.data_type, model_task=ARGS.task, cross_validation=cross_val, 
                                      model_name=ml_name, epoch_batch=epochs_batch,early_stop=early_stop,
                                      features=feature_type,class_weight=cw,encoding_type=encoding_type,
                                        ots_constriants=ots_constraints,transformation=ARGS.transformation,
                                        exclude_guides=ARGS.exclude_guides, test_on_other_data=ARGS.test_on_other_data)
    return file_manager

def init_model_runner(ml_task = None, model_num = None, cross_val = None, features_method = None):
    '''This function creates a run_models object with the given parameters.
    If no parameters are given, the function will use the default parameters passed in the arguments to main.py.
    --------- 
    Returns the run_models object.'''
    from run_models import run_models, tf_clean_up
    atexit.register(tf_clean_up)
    model_runner = run_models()
    if not ml_task: # Not none
        ml_task = ARGS.task
    if not model_num:
        model_num = ARGS.model
    if not cross_val:
        cross_val = ARGS.cross_val
    if not features_method:
        features_method = ARGS.features_method
    model_runner.setup_runner(ml_task=ml_task, model_num=model_num, cross_val=cross_val,
                               features_method=features_method,cw=ARGS.class_weights, encoding_type=ARGS.encoding_type,
                                 if_bulges=with_bulges(ARGS.off_target_constriants), early_stopping=ARGS.early_stoping
                                 ,deep_parameteres=ARGS.deep_params)
    set_multi_process(model_runner.get_gpu_availability())
    return model_runner

def init_model_runner_file_manager(model_params = None):
    if model_params:
        model_runner = init_model_runner(*model_params)
    else :
        model_runner = init_model_runner()
    params = model_runner.get_parameters_by_names()
    file_manager = init_file_management(params)
    model_runner.set_big_wig_number(file_manager.get_number_of_bigiwig())
    
    return model_runner, file_manager

def init_run(model_params = None,  cross_val_params = None):
    '''This function inits the model runner and file manager.
    It sets the cross validation arguments for the file_manager'''
    model_runner, file_manager = init_model_runner_file_manager(model_params)
    train,test,process,evaluation = set_job(ARGS)
    t_guides, ensembles, models = set_cross_val_args(file_manager, train, test, cross_val_params)
    x_features, y_features, all_guides = get_x_y_data(file_manager, model_runner.get_model_booleans())
    return model_runner, file_manager, x_features, y_features, all_guides, t_guides, ensembles, models

def get_x_y_data(file_manager, model_runner_booleans, features_columns = None):
    '''Given a file manager, model runner booleans and model runner codings
    The function will generate the features and labels for the model used by the path to data in the file manager.
    The function will return the x_features, y_features and the guide set.''' 
    
    if_only_seq, if_bp, if_seperate_epi, if_features_by_columns, data_reproducibility, model_repro = model_runner_booleans
    if not features_columns: # if feature columns not given set for the defualt columns
        #### CHANGE to validate column with booleans
        features_columns = ARGS.features_columns
    log_time(f"Features_generation_{encoding_dict()[ARGS.encoding_type]}_start")
    x,y,guides=  generate_features_and_labels(file_manager.get_merged_data_path() , file_manager,
                                         if_bp, if_only_seq, if_seperate_epi,
                                         ARGS.epigenetic_window_size, features_columns, 
                                         data_reproducibility,COLUMNS, ARGS.transformation.lower(),
                                         sequence_coding_type=ARGS.encoding_type, if_bulges= with_bulges(ARGS.off_target_constriants),
                                         exclude_guides = ARGS.exclude_guides,test_on_other_data=file_manager.get_train_on_other_data())
    print(f"Memory Usage features: {get_memory_usage():.2f} MB")
    log_time(f"Features_generation_{encoding_dict()[ARGS.encoding_type]}_end")
    return x,y,guides

def init_evaluation_and_process(base_path, multi_process = False):
    '''This function init an evaluation object and process the scores of the ensemble into a combi file.
    If multi_process is False it will not multi process the inner function used - process_single_ensemble_scores'''
    evaluation_obj = evaluation(ARGS.task)
    evaluation_obj.process_single_ensemble_scores(base_path,multi_process)


def run():
    
    set_args(sys.argv)
    

    global ARGS
    log_time("Main_Run_start")
    train,test,process,evaluation = set_job(ARGS)
    cross_val_dict = cross_val()
    cross_val_dict[ARGS.cross_val](train,test,process,evaluation,ARGS.features_method)
    log_time("Main_Run_end")

def cross_val():
    function_dict = {
        1: run_leave_one_out,
        2: run_k_groups,
        3: run_ensemble,
        4: run_k_groups_with_ensemble,
    }
    return function_dict
def run_ensemble(train = False, test = False,process=False,evaluation=False, method = None):
    if train:
        train_dict = train_ensemble()
        train_dict[method]()
    elif test:
        test_dict = test_ensemble()
        test_dict[method]()
    elif process:
        process_dict = process_ensemble()
        process_dict[method]()
    elif evaluation:
        if ARGS.test_on_other_data:
            evaluate_ensemble_by_guides_in_other_data()
        else:
            evaluate_ensemble_partition()
    else: 
        raise ValueError("Job must be either Train or Test")
   
def train_ensemble():
    return  {
        1: create_ensmble_only_seq,
        2: create_ensembels_by_all_feature_columns,
        
    }
    
def test_ensemble():
    return {
        1: test_ensemble_via_onlyseq_feature,
        2: test_ensemble_by_features,
    }

def process_ensemble():
    return {
        1: process_ensemble_only_seq,
        2: process_ensemble_by_features,
        5: process_ensemble_by_features,
    }

def process_ensemble_only_seq(cross_val_params = None, multi_process = True):
    '''This function process the ensemble scores with only seq features.
    It will init the file manager and set the cross val arguments for the file manager.
    It will use the init evaluation function to process the scores.'''
    file_manager = init_file_management()
    set_cross_val_args(file_manager,test=True,cross_val_params=cross_val_params,cross_val=3)
    ml_results_base_path = file_manager.get_ml_results_path()
    init_evaluation_and_process(ml_results_base_path, multi_process)
    
def process_ensemble_by_features(cross_val_params = None, multi_process = True):
    '''This function process the ensemble scores with other features than only seq.
    The function inits a file manager and get all the features paths by finding the scores and combi folder.
    If there are more than 1 ensemble and multiprocess is allowed, the function will multiprocess the inner function.'''
    file_manager = init_file_management()
    set_cross_val_args(file_manager,test=True,cross_val_params=cross_val_params,cross_val=3)
    ml_results_base_path = file_manager.get_ml_results_path()
    scores_combis_paths = get_scores_combi_paths_for_ensemble(ml_results_base_path,ARGS.n_ensmbels,ARGS.n_models)
    ## if n_ensembles is greater than 1 the multi process will be in the inner function.
    if ARGS.n_ensmbels == 1 and multi_process: # can do multi processing
        processes = min(os.cpu_count(), len(scores_combis_paths))
        with Pool(processes=processes) as pool:
            pool.starmap(init_evaluation_and_process, [(path,) for path in scores_combis_paths])
    else: # n_ensembles > 1
        for path in scores_combis_paths:
            init_evaluation_and_process(path, multi_process)
 
def run_k_groups_with_ensemble(train = False, test = False,process=False,evaluation= False, method = None):
    if train:
        train_dict = train_k_groups_with_ensemble()
        train_dict[method]()
    elif test:
        test_dict = test_k_groups_with_ensemble()
        test_dict[method]()
    elif process:
        process_dict = process_k_groups_with_ensemble()
        process_dict[method]()  
    elif evaluation:
        evaluate_all_partitions()
        
    else: raise ValueError("Job must be either Train/Test/Evaluation")

def evaluate_all_partitions():
    ml_results_base_path,feature_dict,plots_path,partition_info_path,eval_obj = set_evaluation_args()
    eval_obj.evaluate_all_partitions_multiple_metrics((ml_results_base_path,ARGS.partition, ARGS.n_ensmbels, ARGS.n_models,feature_dict,ARGS.other_feature_columns,plots_path,partition_info_path))

def evaluate_ensemble_partition():
    ml_results_base_path,feature_dict,plots_path,partition_info_path,eval_obj = set_evaluation_args()
    eval_obj.evaluate_partitions_ensmbele(ml_results_base_path, ARGS.partition, ARGS.n_ensmbels, ARGS.n_models,feature_dict,ARGS.other_feature_columns,plots_path,partition_info_path)
def evaluate_ensemble_by_guides_in_other_data():
    file_manager = init_file_management()
    ml_results_path = file_manager.get_ml_results_path()
    scores_combi_paths = get_scores_combi_paths_for_ensemble(ml_results_path,ARGS.n_ensmbels,ARGS.n_models,True)
    eval_obj = evaluation(ARGS.task)
    guide_indexes = keep_indexes_per_guide(file_manager.get_merged_data_path(),COLUMNS["TARGET_COLUMN"])
    eval_obj.evaluate_test_per_guide(scores_combi_paths,ARGS.n_ensmbels,ARGS.n_models,guide_indexes,file_manager.get_plots_path())
def set_evaluation_args():
    ARGS.cross_val = 3
    file_manager = init_file_management()
    ml_results_base_path = file_manager.get_ml_results_path()
    # go back to the base path
    ml_results_base_path = os.path.dirname(ml_results_base_path)
    eval_obj = evaluation(ARGS.task)

    feature_dict = split_epigenetic_features_into_groups(ARGS.epi_features_columns)
    plots_path = file_manager.get_plots_path()
    partition_info_path = file_manager.get_partition_information_path()
    with open(ARGS.other_feature_columns, 'r') as f:
        ARGS.other_feature_columns = json.load(f)
    return ml_results_base_path,feature_dict,plots_path,partition_info_path,eval_obj
def process_k_groups_with_ensemble():
    return {
        1: process_k_groups_with_ensemble_only_seq,
        2: process_k_groups_with_ensemble_by_features,
        5: process_k_groups_with_ensemble_by_features,
    }
def process_k_groups_with_ensemble_only_seq():
    ARGS.cross_val = 3
    if ARGS.n_ensmbels == 1: # multi_process each partition
        args = [((ARGS.n_models, ARGS.n_ensmbels, [partition]),False) for partition in ARGS.partition] 
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(process_ensemble_only_seq, args)
    else : # Let the inner function do the multi processing for each ensembles
       for partition in ARGS.partition:
           process_ensemble_only_seq((ARGS.n_models, ARGS.n_ensmbels, [partition]))
def process_k_groups_with_ensemble_by_features():
    ARGS.cross_val = 3
    # Get whats bigger, len of partitions or length of features.
    if ARGS.feature_method == 2:
        if len(ARGS.partition) > len(ARGS.epi_features_columns):
            partition_process = True
    elif ARGS.feature_method == 5:
        if len(ARGS.partition) > len(ARGS.other_feature_columns):
            partition_process = True
    if partition_process:
        args = [((ARGS.n_models, ARGS.n_ensmbels, [partition]),False) for partition in ARGS.partition]
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(process_ensemble_by_features, args)
    else:
        for partition in ARGS.partition:
            process_ensemble_by_features((ARGS.n_models, ARGS.n_ensmbels, [partition]))
 
def train_k_groups_with_ensemble():
    return {
    1: train_k_groups_with_ensemble_only_seq,
    2: train_k_groups_with_ensembles_epi_features,
    5: train_k_groups_with_ensemble_other_features,
    } 
    
def test_k_groups_with_ensemble():
    return {
    1: test_k_groups_with_ensemble_only_seq,
    2: test_k_groups_with_ensembles_epi_features,
    5: test_k_groups_with_ensemble_other_features,
    }
    

def train_k_groups_with_ensemble_only_seq():
    #1. For each partition in partitions need to create n_ensmbels with n_models
    global ARGS
    log_time("Train_k_groups_with_ensemble_only_seq_start")
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False,None, ARGS.features_method)
    ARGS.cross_val = 3 # Set cross val to ensemble
    if ARGS.n_ensmbels == 1 and MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(create_ensmble_only_seq, multi_process_args)
        log_time("Train_k_groups_with_ensemble_only_seq_end-multiprocess")
    else:
        for args in multi_process_args:
            create_ensmble_only_seq(*args)
        log_time("Train_k_groups_with_ensemble_only_seq_end-not_multiprocess")

def test_k_groups_with_ensemble_only_seq():
    global ARGS
    log_time("Test_k_groups_with_ensemble_only_seq_start")
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False, None, ARGS.features_method) # If n_ensembles = 1
    ARGS.cross_val = 3 # Set cross val to ensemble
    if ARGS.n_ensmbels == 1 and MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(test_ensemble_via_onlyseq_feature, multi_process_args)
        log_time("Test_k_groups_with_ensemble_only_seq_end-multiprocess")
    else:
        for args in multi_process_args:
            test_ensemble_via_onlyseq_feature(*args)
        log_time("Test_k_groups_with_ensemble_only_seq_end-not_multiprocess")

def train_k_groups_with_ensembles_epi_features():
    '''This function utilizies the create_ensembels_by_all_epi_feature_columns to train each partition on all epigenetic features.
    It will create different cross_val arguments for each partition and will multiprocess the create function with these parameters.'''
    global ARGS
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False,None, ARGS.features_method)
    ARGS.cross_val = 3
    # For each partition create arguments for the spesific partition. 
    if ARGS.n_ensmbels == 1 and MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(create_ensembels_by_all_feature_columns, multi_process_args)
    else:
        for arg in multi_process_args:
            create_ensembels_by_all_feature_columns(*arg)

def test_k_groups_with_ensembles_epi_features():
    global ARGS
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False,None, ARGS.features_method)
    ARGS.cross_val = 3
    if ARGS.n_ensmbels == 1 and MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(test_ensemble_by_features, multi_process_args)
    else:
        for arg in multi_process_args:
            test_ensemble_by_features(*arg)
##### NOTE: MAYBE REMOVE THIS FUNCTION WHERE ALL FEATURES ARE ADDED IN ONE FUNCTION #### 
def train_k_groups_with_ensemble_other_features():
    ARGS.cross_val = 3
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False,ARGS.other_feature_columns, ARGS.features_method)
    if ARGS.n_ensmbels == 1:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(create_ensembels_by_all_feature_columns, multi_process_args)
    else:
        for arg in multi_process_args:
            create_ensembels_by_all_feature_columns(*arg)

def test_k_groups_with_ensemble_other_features():
    ARGS.cross_val = 3
    multi_process_args = get_k_groups_ensemble_args(ARGS.partition, ARGS.n_models, ARGS.n_ensmbels,False,ARGS.other_feature_columns, ARGS.features_method)
    if ARGS.n_ensmbels == 1:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(test_ensemble_by_features, multi_process_args)
    else:
        for arg in multi_process_args:
            test_ensemble_by_features(*arg)

#####################################################################
   
def run_k_groups(train = False, test = False,process=False,evaluation=False, method = None):
    if train:
        train_dict = train_k_groups()
        train_dict[method]()
    elif test:
        test_dict = test_k_groups()
        test_dict[method]()

def train_k_groups():
    return {
        1: train_k_groups_only_seq,
        2: train_k_groups_by_features,
    }
def test_k_groups():
    return {
        1: test_k_groups_only_seq,
        2: test_k_groups_by_features,
    }
def test_k_groups_by_features():
    pass
def run_leave_one_out(train = False, test = False):
    pass
def train_k_groups_by_features():
    runner,file_manager = init_model_runner_file_manager()
    train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False)
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()

def train_k_groups_only_seq():
    '''
    This function trains a model for each partition.
    In total k models will be created with the suffix partition_number.keras
    '''
    runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run()
    models_path = file_manager.get_model_path()
    seed=10
    args = [(os.path.join(models_path, f"{partition}.keras"),guides[partition],seed,x_features,y_features,all_guides )for partition in ARGS.partition]
    if MULTI_PROCESS:
        processes = min(os.cpu_count(), len(ARGS.partition))
        with Pool(processes=processes) as pool:
            pool.starmap(runner.create_model, args)
    else:
        for arg in args:
            runner.create_model(*arg)       
def test_k_groups_only_seq(if_plot = True):
    '''
    This functions tests every model in k_cross partition and calculate it evaluation metric.
    NOTE: ADD ARGUMENTS FOR FEATURES AND MULTIPROCESSING
    '''
    runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run()
    models_path = file_manager.get_model_path()
    scores_dictionary = {}
    if ARGS.test_on_other_data: # Test on other data so 
        models = create_paths(models_path)
        for model in models:
            scores, test, idx = runner.test_model(model, guides, x_features, y_features, all_guides)
            scores_dictionary[model] = (scores,test,idx)
    else:
        for partition in ARGS.partition:
            temp_path = os.path.join(models_path, f"{partition}.keras")
            scores, test, idx = runner.test_model(temp_path, guides[partition], x_features, y_features, all_guides)
            scores_dictionary[partition] = (scores,test,idx)
    evaluation_obj = evaluation(ARGS.task)
    results = evaluation_obj.get_k_groups_results(scores_dictionary)
    ml_results = file_manager.get_ml_results_path()
    results.to_csv(os.path.join(ml_results, "results.csv"), index=False)
    if if_plot:
        plots_path = file_manager.get_plots_path()
        evaluation_obj.plot_k_groups_results(scores_dictionary,plots_path, features_method_dict()[ARGS.features_method])
        
    
def train_leave_one_out():
    pass
def test_leave_one_out():
    pass




## ENSMBEL
# Only sequence
def create_ensmble_only_seq(  model_params = None,cross_val_params=None,multi_process=True,group_dir = None,):
    '''The function will create an ensmbel with only sequence features'''
    log_time("Create_ensmble_only_seq_start")
    if not model_params and not cross_val_params: # None
        runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run()
    else:
        runner, file_manager , x_features, y_features, all_guides, guides, n_models, n_ensmbels = init_run(model_params, cross_val_params)
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    if n_ensmbels == 1:
        create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner, x_features, y_features, all_guides)
    else: # more then 1 ensembles.
        create_n_ensembles(n_ensmbels, n_models, guides, file_manager, runner, x_features, y_features, all_guides, multi_process=True)
    log_time("Create_ensmble_only_seq_end")
    del x_features, y_features
## - EPIGENETICS:
    ## 1. Creation
def create_ensembels_by_all_feature_columns(model_params = None,cross_val_params=None, multi_process = True, feature_dict = None):
    '''
    This function trains ensembles for each epigenetic feature column.
    The function splits the columns them into groups of the epigenetic value estimation i.e binary, score, enrichment.
    The function will create ensmbels for each group and for each feature in the group.
    NOTE: multiprocess each feature when n_ensmbels = 1. if n_ensembles > 1 the subfunction will multiprocess the ensmbels.
    ARGS:
    1. model_params: tuple - model parameters for the runner
    2. cross_val_params: tuple - cross val parameters for the file manager
    3. multi_process: bool - if True the function will multiprocess the features in the group.'''
    if not feature_dict: # None
        ### NOTE: ONLY EPIGENETICS IS SET TO TRUE!!!
        features_dict = parse_feature_column_dict(ARGS.features_columns, only_epigenetics=True)
    else:
        features_dict = feature_dict
    arg_list = []
    if not model_params and not cross_val_params: # No paramaeteres are given.
        runner, file_manager = init_model_runner_file_manager()
        train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False)
    else:
        runner, file_manager = init_model_runner_file_manager(model_params)
        train_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = True, test = False, cross_val_params = cross_val_params)
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    
    if n_ensmbels == 1 and multi_process and MULTI_PROCESS:  # multiprocess bool for activating this function from another function.
        arg_list = get_features_columns_args_ensembles(runner= runner, file_manager = file_manager, t_guides = train_guides, 
                                             model_base_path = model_base_path, ml_results_base_path = ml_results_base_path,
                                               n_models = n_models, n_ensmbels = n_ensmbels, features_dict = features_dict, multi_process = False)
        with Pool(processes=10) as pool:
            pool.starmap(create_ensembels_for_a_given_feature, arg_list) 
    else: # multi_process = True/False and n_ensmbels > 1
        arg_list = get_features_columns_args_ensembles(runner= runner, file_manager = file_manager, t_guides = train_guides, 
                                             model_base_path = model_base_path, ml_results_base_path = ml_results_base_path,
                                               n_models = n_models, n_ensmbels = n_ensmbels, features_dict = features_dict, multi_process = multi_process)
        for args in arg_list:
            create_ensembels_for_a_given_feature(*args)
            



def create_ensembels_for_a_given_feature(group, feature,runner, file_manager,train_guides,model_base_path,ml_results_base_path,n_models=50, n_ensmbels=10, multi_process = False):
    '''This function create a ensemble for a given group of features and A feature in that group by utilizing the create_n_ensembles function.
    It extracts the x_features, y_features and all_guides from the file manager given the specific feature and create the ensembles.
    ARGS:
    1. group: str - group name
    2. feature: list of one feature
    3. runner: run_models object
    4. file_manager: file_manager object
    5. train_guides: list of guides to train the model on
    6. model_base_path: str - model path before the group and feature
    7. ml_results_base_path: str - ml results path " " " " " " " " ..
    8. n_models: int - number of models in each ensemble
    9. n_ensmbels: int - number of ensembles
    10. multi_process: bool - passed to create_n_ensembles function to multiprocess the ensembles.'''
    log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_start')
    x_features,y_features,all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
    temp_suffix = get_feature_column_suffix(group,feature) # set path to epigenetic data type - binary, by score, by enrichment.
    file_manager.set_models_path(model_base_path) # set model path
    file_manager.set_ml_results_path(ml_results_base_path) # set ml results path
    file_manager.add_type_to_models_paths(temp_suffix) # add path to train ensmbel
    runner.set_features_columns(feature) # set feature   
    create_n_ensembles(n_ensmbels, n_models, train_guides, file_manager, runner,x_features,y_features,all_guides, multi_process)
    log_time(f'Create_ensmbels_with_epigenetic_features_{group}_{feature}_end')
    # Delete data free memory
    del x_features, y_features
def create_n_ensembles(n_ensembles, n_models, guides, file_manager, runner, x_, y_, all_guides, multi_process = False, start_from = 3): 
    '''This function creates n ensembles with n models for each ensemble.
    It will use the file manager to create train folders for each ensmbel.
    It will use the model runner to train the model in that folder.
    ARGS:
    1. n_ensembles: int - number of ensembles
    2. n_models: int - number of models in each ensemble
    3. guides: list of guides to train the model on
    4. file_manager: file_manager object
    5. runner: run_models object
    6. x_: np.array - features
    7. y_: np.array - labels
    8. all_guides: list of all guides in the data
    9. multi_process: bool - if True and the number of ensmebles is bigger than 1, the function will multiprocess the ensembles.'''
    # Generate argument list for each ensemble
    ensemble_args_list = [(n_models, file_manager.create_ensemble_train_folder(i), guides,(i*10),x_,y_,all_guides) for i in range(start_from, n_ensembles+1)]
    # Create_ensmbel accpets - n_models, output_path, guides, additional_seed for reproducibility
    if multi_process and n_ensembles > 1 and MULTI_PROCESS:
        # Create a pool of processes
        cpu_count = os.cpu_count()
        num_proceses = min(cpu_count, n_ensembles)
        with Pool(processes=num_proceses) as pool:
            pool.starmap(runner.create_ensemble, ensemble_args_list)
    else : 
        for args in ensemble_args_list:
            runner.create_ensemble(*args)


## 2. ENSMBEL SCORES/Predictions


def test_ensemble_via_onlyseq_feature(model_params = None,cross_val_params=None,multi_process=True,different_test_folder_path = None, different_test_path = None, group_dir = None):
    '''This function init a model runner, file manager and the x,y features and testing guide for an ensmeble.
    It will pass these arguments to test_ensmbel_scores function to test the ensmbel on the test guides.
    If more than 1 ensemble to check, the function will multiprocess if it wasnt activated from a multiprocess program.
    ARGS:
    1. model_params: tuple - model parameters for the runner
    2. cross_val_params: tuple - cross val parameters for the file manager
    3. multi_process: bool set to True, when another process subprocess this function it should set to False to avoid sub multiprocessing.'''
    if not model_params and not cross_val_params: # None
        runner, file_manager , x_features, y_features, all_guides, tested_guides, n_models, n_ensmbels = init_run()
    else :
        runner, file_manager , x_features, y_features, all_guides, tested_guides, n_models, n_ensmbels = init_run(model_params, cross_val_params)
    if different_test_folder_path: # Not None
        file_manager.set_ml_results_path(different_test_folder_path)
    if different_test_path:
        file_manager.set_seperate_test_data(different_test_path[0],different_test_path[1])    
    if group_dir:
        file_manager.add_type_to_models_paths(group_dir)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, tested_guides, score_path, x_features, y_features, all_guides) for ensmbel in ensmbels_paths]
    if n_ensmbels>1 and multi_process and MULTI_PROCESS: 
        with Pool(processes=10) as pool:
            pool.starmap(test_enmsbel_scores, args)
    else:
        for ensmbel in ensmbels_paths:
            test_enmsbel_scores(runner, ensmbel, tested_guides, score_path, x_features, y_features, all_guides)


def test_enmsbel_scores(runner, ensmbel_path, test_guides, score_path, x_features, y_labels, all_guides):
    '''Given a path to an ensmbel, a list of test guides and a score path
    the function will test the ensmbel on the test guides and save the scores in the score path.
    Each scores will be added with the acctual label and the index of the data point.'''
    
    print(f"Testing ensmbel {ensmbel_path}")
    models_path_list = create_paths(ensmbel_path)
    models_path_list.sort(key=lambda x: int(x.split(".")[-2].split("_")[-1]))  # Sort model paths by models number
    y_scores, y_test, test_indexes = runner.test_ensmbel(models_path_list, test_guides, x_features, y_labels, all_guides)
    # Save raw scores in score path
    temp_output_path = os.path.join(score_path,f'{ensmbel_path.split("/")[-1]}.csv')
    y_scores_with_test = add_row_to_np_array(y_scores, y_test)  # add accual labels to the scores
    y_scores_with_test = add_row_to_np_array(y_scores_with_test, test_indexes) # add the indexes of each data point
    y_scores_with_test = y_scores_with_test[:,y_scores_with_test[-1,:].argsort()] # sort by indexes
    write_2d_array_to_csv(y_scores_with_test,temp_output_path,[])






def test_ensemble_by_features(model_params= None, cross_val_params= None, multi_process = True, features_dict = None):
    if not model_params and not cross_val_params: # None
        runner, file_manager  = init_model_runner_file_manager()
        t_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True)
    else : 
        runner, file_manager  = init_model_runner_file_manager(model_params)
        t_guides, n_models, n_ensmbels = set_cross_val_args(file_manager, train = False, test = True, cross_val_params = cross_val_params)
    if not features_dict: # None
        features_dict = parse_feature_column_dict(ARGS.features_columns,only_epigenetics=True)
    else:
        features_dict = features_dict
    
    model_base_path, ml_results_base_path = file_manager.get_model_path(), file_manager.get_ml_results_path()
    arg_list = []
    if n_ensmbels == 1 and multi_process and MULTI_PROCESS: # multiprocess each feature
        arg_list = get_features_columns_args_ensembles(runner, file_manager, t_guides, model_base_path, ml_results_base_path, n_models, n_ensmbels, features_dict, multi_process = False)
        with Pool(processes=10) as pool:
            pool.starmap(test_ensemble_via_epi_feature_2, arg_list)
    else:
        arg_list = get_features_columns_args_ensembles(runner, file_manager, t_guides, model_base_path, ml_results_base_path, n_models, n_ensmbels, features_dict, multi_process = multi_process)
        for arg in arg_list:
            test_ensemble_via_epi_feature_2(*arg)
    
def test_ensemble_via_epi_feature_2(group, feature, runner, file_manager, t_guides, model_base_path, ml_results_base_path,n_models, n_ensmbels, multi_process):
    
    group_epi_path = get_feature_column_suffix(group,feature)
    file_manager.set_models_path(model_base_path)
    file_manager.set_ml_results_path(ml_results_base_path)
    file_manager.add_type_to_models_paths(group_epi_path)
    runner.set_features_columns(feature)
    x_features, y_features, all_guides = get_x_y_data(file_manager, runner.get_model_booleans(),  feature)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder() # Create score and combi folders
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, t_guides, score_path,x_features,y_features,all_guides) for ensmbel in ensmbels_paths]
    if multi_process and MULTI_PROCESS: # can run multiprocess
        with Pool(processes=10) as pool:
            pool.starmap(test_enmsbel_scores, args)
    else: 
        for arg in args:
            test_enmsbel_scores(*arg)
    del x_features, y_features




def set_reproducibility_data(file_manager, run_models, data_path):
    file_manager.ml_results_path(data_path)
    run_models.set_data_reproducibility(True)
    run_models.set_model_reproducibility(False)
def set_reproducibility_models(file_manager, run_models, model_path):
    file_manager.ml_results_path(model_path)
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


if __name__ == "__main__":
    
    set_time_log(keep_time=True,time_logs_paths="/home/dsi/lubosha/Off-Target-data-proccessing/Time_logs")
    
    try:
        ## NOTE: CHANGE START_FROM IN CREATE ENSEMBLE
        run()
    except Exception as e:
        print(e)
    #evalaute_all_partitions()
    # partition_data_for_histograms("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/vivovitro_nobulges_withEpigenetic_indexed_read_count_with_model_scores.csv",
    #                               partition_information_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/partition_guides_78/Changeseq-Partition_vivo_vitro.csv",
    #                               data_name="Change-seq",data_type="vivo-vitro",off_target_constraints=2,partition_number=1,
    #                               output_path="/home/dsi/lubosha/Off-Target-data-proccessing/Partition_analysis",target_column="target",
    #                               label_column="Label",bulge_column=None,mismatch_column="distance")
                             
    save_log_time(ARGS)
    
    #performance_by_increasing_positives("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv","/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive","")
    
    # test_performance_by_data(Models_folder="/localdata/alon/Models/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positive",
    #                          test_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism.csv",
    #                          test_guides="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Partitions_guides/tested_guides_12_partition.txt")
    #combiscore_by_folder("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives")
    # bar_plot_ensembels_feature_performance(only_seq_combi_path="/localdata/alon/ML_results/Change-seq/Train_vitro_test_genome/CNN/Ensemble/Only_sequence/1_partition/1_partition_50/Combi",
    #                                        epigenetics_path="/localdata/alon/ML_results/Change-seq/Train_vitro_test_genome/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary",
    #                                        n_models_in_ensmbel=50,output_path="/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq/train_vitro_test_silico",
    #                                        title="Train_vitro_test_silico")
    
 
    #process_score_path("/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives/_1/Scores/_1.csv","/localdata/alon/ML_results/Hendel/vivo-silico/Performance-by-data/CNN/Ensemble/Only_sequence/by_positives/_1/Combi")
   