from Server_constants import   EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_GS_EPI, DATA_PATH
from file_management import File_management
from run_models import run_models

def init_file_management():
    return File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_GS_EPI, DATA_PATH)
def init_run_models(file_manager):
    return run_models(file_manager)
def main():
    file_manager = init_file_management()
    run_models = init_run_models(file_manager)
    run_models.run("Change_csgs")

   

if __name__ == "__main__":
    main()