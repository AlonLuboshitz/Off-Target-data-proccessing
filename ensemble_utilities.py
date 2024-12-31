'''
Module for ensemble utilities.
'''
from file_utilities import find_target_folders

def get_scores_combi_paths_for_ensemble(ml_results_path,n_ensembles,n_modles, all_features=False  ):
    '''
    This function extracts from the ml_results_path paths that contianing the scores and combi folder 
    indicating there is a model that was trained and tested.
    Args:
    1. ml_results_path - (str) path to the ml_results folder
    2. n_ensembles - (int) number of ensembles
    3. n_modles - (int) number of models
    4. all_features - (bool) if True, will return all the paths that contain the n_ensembles and n_models
    --------
    Returns: list of paths to the scores and combi folders.'''
    if all_features:
        ml_results_path = ml_results_path.split("Ensemble")[0]
    scores_combis_paths =  find_target_folders(ml_results_path, ["Scores", "Combi"])
    # remove folder that dont hold the n_ensembles and n_models
    for path in scores_combis_paths:
        if f'{n_ensembles}_ensembels' not in path or f'{n_modles}_models' not in path:
            scores_combis_paths.remove(path)
    return scores_combis_paths