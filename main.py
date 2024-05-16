from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_CASO_EPI,CHANGESEQ_GS_EPI
from Server_constants import DATA_PATH, DATA_REPRODUCIBILITY, MODEL_REPRODUCIBILITY, ALL_REPRODUCIBILITY
from Server_constants import  ENSMBEL_GUIDES_FOLDER
from Server_constants import ENSEMBEL_MODELS_FOLDER_LOCAL, ENSEMBEL_RESULTS_FOLDER_LOCAL
from Server_constants import CS_MODEL_PATH, CS_ML_RESULTS
from constants import FEATURES_COLUMNS
from multiprocessing import Pool

from file_management import File_management
#from run_models import run_models
from utilities import set_epigenetic_features_by_string, split_epigenetic_features_into_groups, create_guides_list, write_2d_array_to_csv, create_paths, keep_only_folders, add_row_to_np_array, extract_scores_labels_indexes_from_files
from evaluation import eval_all_combinatorical_ensmbel
import os

## INIT FILE MANAGER
def init_file_management():
    file_manager = File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_GS_EPI , DATA_PATH)
    set_file_manager_files(file_manager)
    return file_manager
def set_file_manager_files(file_manager):
    file_manager.set_ml_results_path(ENSEMBEL_RESULTS_FOLDER_LOCAL)
    file_manager.set_models_path(ENSEMBEL_MODELS_FOLDER_LOCAL)
    file_manager.set_ensmbel_guides_path(ENSMBEL_GUIDES_FOLDER)

def set_file_manager_ensmbel_files(file_manager):
    file_manager.set_ensmbel_train_path(ENSEMBEL_MODELS_FOLDER_LOCAL)
    file_manager.set_ensmbel_guides_path(ENSMBEL_GUIDES_FOLDER)
    file_manager.set_ensmbel_result_path(ENSEMBEL_RESULTS_FOLDER_LOCAL)
    
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
    
    file_manager.set_partition(int(partition_num))
    # Get guides
    guides_path = file_manager.get_guides_partition()
    guides = []
    if partition_num == 0: # All guides
        for guide_path in guides_path:
            guides += create_guides_list(guide_path, 0) 
    else: guides = create_guides_list(guides_path, 0)

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
def create_ensmble_only_seq():
    '''The function will create an ensmbel with only sequence features'''
    runner, file_manager = set_up_model_runner()
    init_cnn(runner)
    init_ensmbel(runner)
    init_only_seq(runner)
    runner.setup_runner()
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager,n_models=50, n_ensmbels=10, partition_num=0)
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
    from multiprocessing import Pool
    cpu_count = os.cpu_count()
    num_proceses = min(cpu_count, n_ensembles)
    # Generate argument list for each ensemble
    ensemble_args_list = [(n_models, file_manager.create_ensemble_train_folder(i), guides,(i*10)) for i in range(1, n_ensembles+1)]
    # Create_ensmbel accpets - n_models, output_path, guides, additional_seed for reproducibility
    # Create a pool of processes
    
    with Pool(processes=num_proceses) as pool:
        # Use starmap to map the worker function to the arguments list
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
   

def test_ensemble_via_onlyseq_feature(n_models=50,different_test_folder_path = None):
    runner, file_manager = set_up_model_runner()
    if different_test_folder_path: # Not None
        file_manager.set_ml_results_path(different_test_folder_path)
    init_cnn(runner)
    init_ensmbel(runner)
    init_only_seq(runner)
    runner.setup_runner()
    guides, n_models, n_ensmbels = set_ensmbel_preferences(file_manager, n_models=n_models, n_ensmbels=10, partition_num=0)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    ensmbels_paths = create_paths(file_manager.get_model_path())  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    args = [(runner, ensmbel, guides, score_path) for ensmbel in ensmbels_paths]
    with Pool(processes=10) as pool:
        pool.starmap(test_enmsbel_scores, args)
    
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
def process_single_ensemble_scores(scores_path):
    '''This function will process all the ensmbel scores in the given path
Given a score csv file it will extract from the scores diffrenet combinations of the scores and evaluate them 
vs the labels. The results will be saved in the combi path for the same ensmbel.'''
    
    file_manager = init_file_management()
    file_manager.set_ml_results_path(scores_path)
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder()
    
    ensmbel_scores_paths = create_paths(score_path) # Get a list of paths for each ensmbel scores
   
    # Number of processes in the pool
    num_cores = os.cpu_count()
    num_processes = min(num_cores, len(ensmbel_scores_paths))
    # Add the combi path to ensmbel paths for the process function
    ensmbel_scores_paths = [(score_path, combi_path) for score_path in ensmbel_scores_paths]
    # Create a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        # Map the function to the list of paths
        pool.starmap(process_score_path, ensmbel_scores_paths)
        

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
    set_reproducibility_data(file_manager, run_models, DATA_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)
def run_reproducibility_models(run_models, model_name, file_manager, k_times):
    set_reproducibility_models(file_manager, run_models, MODEL_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)
def run_reproducibility_model_and_data(run_models, model_name, file_manager, k_times):
    set_reproducibility_data(file_manager, run_models, DATA_REPRODUCIBILITY)
    set_reproducibility_models(file_manager, run_models, ALL_REPRODUCIBILITY)
    for i in range(k_times):
        temp_model_name = f"{model_name}_{i}"
        run_model_only_seq(run_models, temp_model_name)


### With creation of ensemble i seed is *100 and not 10! need to change back!
def only_seq_ensemble_pipe():
    create_ensmble_only_seq()
    #test_ensemble_via_onlyseq_feature(n_models=50,different_test_folder_path="/localdata/alon/ML_results/Train_vitro_test_genome")
    #process_single_ensemble_scores("/localdata/alon/ML_results/Train_vitro_test_genome/CNN/Ensemble/Only_sequence/1_partition/1_partition_50")
    pass
def epigeneitc_ensemble_pipe():
    from Server_constants import LOCAL_RESULTS_EPIGENETICS, LOCAL_MODELS_EPIGENETICS
    #create_ensembels_by_feature_columns()
    #test_ensembles_via_epi_features_in_folder(LOCAL_MODELS_EPIGENETICS,"/localdata/alon/ML_results/Train_vitro_test_genome")
    process_all_ensembels_scores_in_folder("/localdata/alon/ML_results/Train_vitro_test_genome/CNN/Ensemble/Epigenetics_by_features/1_partition/1_partition_50/binary")
    pass


if __name__ == "__main__":
    #epigeneitc_ensemble_pipe()
    only_seq_ensemble_pipe()
