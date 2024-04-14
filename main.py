from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_CASO_EPI,CHANGESEQ_GS_EPI, DATA_PATH, DATA_REPRODUCIBILITY, MODEL_REPRODUCIBILITY, ALL_REPRODUCIBILITY
from file_management import File_management
#from run_models import run_models
from utilities import split_epigenetic_features_into_groups, create_guides_list, write_2d_array_to_csv, create_paths, keep_only_folders, add_row_to_np_array, extract_scores_labels_indexes_from_files
from evaluation import eval_all_combinatorical_ensmbel
import os

def init_file_management():
    return File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_GS_EPI , DATA_PATH)
def init_run_models(file_manager):
    from run_models import run_models
    model_runner = run_models(file_manager)
    model_runner.setup_runner()
    return model_runner
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

'''Given the number of models to run, function will init an file manager
set an enmsbel output and create a list of guides from the guides file via the given list index
then it will create an output path and run N models without the tested guides and save them into the path.'''
def test_ensmbel_partition():
    from run_models import run_models
    from Server_constants import ENSMBEL, ENSMBEL_GUIDES, ENSMBEL_RESULTS
    file_manager = init_file_management()
    file_manager.set_ensmbel_result_path(ENSMBEL_RESULTS) # Set ensmbel results path
    score_path, combi_path = file_manager.create_ensemble_score_nd_combi_folder() # Create score and combi folders
    ensmbels_paths = create_paths(ENSMBEL)  # Create paths for each ensmbel in partition
    ensmbels_paths = keep_only_folders(ensmbels_paths)  # Keep only folders
    guides = create_guides_list(ENSMBEL_GUIDES, 0)  # Create guides list    
    runner = run_models(file_manager) # Init runner
    runner.setup_ensmbel_runner() # Setup runner
    runner.set_features_columns(["Chromstate_atacseq_peaks_binary","Chromstate_h3k4me3_peaks_binary"]) # Set features columns ,""
    
    for ensmbel in ensmbels_paths:
        print(f"Testing ensmbel {ensmbel}")
        test_enmsbel_scores(runner, ensmbel, guides, score_path)
       
def test_enmsbel_scores(runner, ensmbel_path, test_guides, score_path):
    models_path_list = create_paths(ensmbel_path)
    models_path_list.sort(key=lambda x: int(x.split(".")[-2].split("_")[-1]))  # Sort model paths by models number
    y_scores, y_test, test_indexes = runner.test_ensmbel(models_path_list, test_guides)
    # Save raw scores in score path
    temp_output_path = os.path.join(score_path,f'{ensmbel_path.split("/")[-1]}.csv')
    y_scores_with_test = add_row_to_np_array(y_scores, y_test)  # add accual labels to the scores
    y_scores_with_test = add_row_to_np_array(y_scores_with_test, test_indexes) # add the indexes of each data point
    y_scores_with_test = y_scores_with_test[:,y_scores_with_test[-1,:].argsort()] # sort by indexes
    write_2d_array_to_csv(y_scores_with_test,temp_output_path,[])




# Function to process a single score path
def process_score_path(score_path,combi_path):
    y_scores, y_test, indexes = extract_scores_labels_indexes_from_files([score_path])
    results = eval_all_combinatorical_ensmbel(y_scores, y_test)
    header = ["Auroc", "Auprc", "N-rank", "Auroc_std", "Auprc_std", "N-rank_std"]
    temp_output_path = os.path.join(combi_path, f'{score_path.split("/")[-1]}')
    write_2d_array_to_csv(results, temp_output_path, header)
'''This function will process all the ensmbel scores in the given path
Given a score csv file it will extract from the scores diffrenet combinations of the scores and evaluate them 
vs the labels. The results will be saved in the combi path for the same ensmbel.'''
def process_ensmbel_scores():
    from multiprocessing import Pool
    from Server_constants import ENSMBEL_RESULTS
    file_manager = init_file_management()
    file_manager.set_ensmbel_result_path(ENSMBEL_RESULTS)
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
 
'''Given number of ensembls to create and n_models to create in each ensmbel
it will create n_ensembles of n_models'''
def create_n_ensembles(n_ensembles, n_modles):
    from run_models import run_models
    from Server_constants import ENSMBEL, ENSMBEL_GUIDES
    file_manager = init_file_management()
    file_manager.set_ensmbel_train_path(ENSMBEL)
    guides = create_guides_list(ENSMBEL_GUIDES, 0)
    runner = run_models(file_manager)
    runner.setup_ensmbel_runner()
    runner.set_features_columns(["Chromstate_atacseq_peaks_binary","Chromstate_h3k4me3_peaks_binary"]) #""Chromstate_atacseq_peaks_binary]
    for i in range(1,n_ensembles+1):
        output_path = file_manager.create_ensemble_train_folder(i)
        runner.create_ensemble(n_modles, output_path, guides)

def create_ensembels_with_epigenetic_features():
    # set path to epigenetic data type - binary, by score, by enrichment.
    # Data partition
    # 1. set path
    # 2. set guides
    # 3. set models amount in each ensmbel
    # Epitgentic features:
    # 1. Split features into groups - binary, by score, by enrichment
    # 2. match the epigenetic data type to the group
    # 3. for each individiual epigenetic mark create a folder
    # 4. create ensmbel in that folder with that epigenetic mark
    # do 3 and 4 for all marks togther.
    # do 3 and 4 for chosen marks togther.
def main():
    create_n_ensembles(10,80)
    

   

if __name__ == "__main__":
    main()