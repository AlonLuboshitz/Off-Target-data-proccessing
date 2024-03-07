from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_CASO_EPI,CHANGESEQ_GS_EPI, DATA_PATH, DATA_REPRODUCIBILITY, MODEL_REPRODUCIBILITY, ALL_REPRODUCIBILITY
from file_management import File_management
from run_models import run_models
from utilities import validate_dictionary_input, create_guides_list, write_2d_array_to_csv
from evaluation import eval_all_combinatorical_ensmbel
import os

def init_file_management():
    return File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_GS_EPI , DATA_PATH)
def init_run_models(file_manager):
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

    
    

def test_enmsbel():
    from Server_constants import ENSMBEL, ENSMBEL_GUIDES, ENSMBEL_RESULTS
    file_manager = init_file_management()
    file_manager.set_ensmbel_path(ENSMBEL)
    models_path_list = file_manager.get_ensmbel_models_path_list()
    guides = create_guides_list(ENSMBEL_GUIDES, 0)
    runner = run_models(file_manager)
    runner.setup_ensmbel_runner()
    y_scores, y_test = runner.test_ensmbel(models_path_list, guides)
    header = ["Auroc","Auprc","N-rank","Auroc_std","Auprc_std","N-rank_std"]
    results = eval_all_combinatorical_ensmbel(y_scores, y_test)
    # add ensmbel number to ensmbel results path
    output_path = os.path.join(ENSMBEL_RESULTS,f'{ENSMBEL.split("/")[-1]}.csv')
    write_2d_array_to_csv(results, output_path, header)
'''Given number of model for each ensmbel and the amount of ensembels create N ensembels with N models'''
def create_n_ensembles(n, n_modles):
    from Server_constants import ENSMBEL, ENSMBEL_GUIDES
    file_manager = init_file_management()
    file_manager.set_ensmbel_output_path(ENSMBEL)
    guides = create_guides_list(ENSMBEL_GUIDES, 0)
    runner = run_models(file_manager)
    runner.setup_ensmbel_runner()
    for i in range(2,n+1):
        output_path = file_manager.create_ensemble_train_folder(i)
        runner.create_ensemble(n_modles, output_path, guides)
        
def main():
    

    create_n_ensembles(10, 10)
    

   

if __name__ == "__main__":
    main()