from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_CASO_EPI,CHANGESEQ_GS_EPI, DATA_PATH, DATA_REPRODUCIBILITY, MODEL_REPRODUCIBILITY, ALL_REPRODUCIBILITY
from file_management import File_management
from run_models import run_models

def init_file_management():
    return File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_CASO_EPI , DATA_PATH)
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
def main():
    file_manager = init_file_management()
    run_models = init_run_models(file_manager)
    #run_reproducibility_model_and_data(run_models, "Change_csgs", file_manager, 2)
    #run_reproducibility_data(run_models, "Change_caso", file_manager, 10)
    run_reproducibility_models(run_models, "Change_caso", file_manager, 10)
    

   

if __name__ == "__main__":
    main()