# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
# "Chromstate_atacseq_peaks_score","Chromstate_atacseq_peaks_fold_enrichemnt","Chromstate_h3k4me3_peaks_score","Chromstate_h3k4me3_peaks_fold_enrichemnt"
FEATURES_COLUMNS = ["Chromstate_atacseq_peaks_binary","Chromstate_h3k4me3_peaks_binary"]
ML_TYPE = "" # String inputed by user
SHUFFLE = True
IF_OS = False
BP_PRESENTATION = 6
GUIDE_LENGTH = 23
ENCODED_LENGTH =  GUIDE_LENGTH * BP_PRESENTATION
IF_SEPERATE_EPI = False
FORCE_USE_CPU = False
EPIGENETIC_WINDOW_SIZE = 0

FIT_PARAMS = {
    'epochs': 5,
    'batch_size': 1024,
    'verbose' : 2,

    # Add any other fit parameters you need
}

from file_management import File_management
from common_variables import common_variables
from features_engineering import generate_features_and_labels, order_data, get_tp_tn
from models import get_cnn, get_logreg, get_xgboost, get_xgboost_cw
import pandas as pd
import numpy as np
import sys
import time
from imblearn.over_sampling import RandomOverSampler, SMOTE
import logging
import os
if FORCE_USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"  
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)







'''function write to table by columsn:
ML type, auroc, auprc from log_reg, unpacks 4 element tuple - tp,tn test, tp,tn train.
features included for training the model
what file was left out.'''
def write_to_table(auroc,auprc,file_left_out,table,ML_type,Tpn_tuple,n_rank):
    global FEATURES_COLUMNS
    features_columns = FEATURES_COLUMNS.copy()
    if ONLY_SEQ_INFO:
        features_columns = ["Only_Seq"]
    try:
        new_row_index = len(table)  # Get the index for the new row
        table.loc[new_row_index] = [ML_type, auroc, auprc,*n_rank,*Tpn_tuple, features_columns, file_left_out]  # Add data to the new row
    except: # empty data frame
        table.loc[0] = [ML_type, auroc, auprc,*Tpn_tuple , features_columns, file_left_out]
    return table
def create_file_name(ending,file_manager,sampler,common_variables_ins):
    features_columns = common_variables_ins.get_features_columns()
    if common_variables_ins.get_if_only_seq():
        features_columns = ["Only_Seq"]
    elif common_variables_ins.get_if_bp():
        features_columns = [file_name[0] for file_name in file_manager.get_bigwig_files()]
    elif common_variables_ins.get_if_seperate_epi():
        features_columns = [file_name[0] for file_name in file_manager.get_bigwig_files()]
        features_columns.append(f'window_{common_variables_ins.get_window_size()}')

    feature_str = ""
    for feature in features_columns:
      feature_str = feature_str + "_" + feature
    sampler_str = f'{get_sampler_type(sampler)}_'
    file_name = f'{ML_TYPE}_{sampler_str}{feature_str}_{ending}.csv'
    return file_name

def crisprsql(data_table,file_name,file_manager,sampler,common_variables_ins):
    x_feature,y_label,guides = generate_features_and_labels(data_table=data_table, manager=file_manager,common_variables=common_variables_ins) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
    print("staring ml")
    file_name = create_file_name(file_name,file_manager,sampler,common_variables_ins)
    for i,key in enumerate(guides):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i,if_shuffle=True,if_print=False,sampler=sampler,if_over_sample=common_variables_ins.get_if_over_sample())
        auroc,auprc,y_score = get_ml_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        n_rank_score = get_auc_by_tpr(tpr_arr=get_tpr_by_n_expriments(predicted_vals=y_score,y_test=y_test,n=1000))
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        print(f"Ith: {i+1}\{len(guides)} split is done")
        results_table = write_to_table(auroc=auroc,auprc=auprc,file_left_out=key,table=results_table,ML_type=ML_TYPE,Tpn_tuple=tps_tns,n_rank=n_rank_score)
        if auroc <= 0.5:
            write_scores(key,y_test,y_score,file_name,auroc)            
    results_table = results_table.sort_values(by="File_out")
    results_table.to_csv(file_name)
    
'''write score vs test if auc<0.5'''
def write_scores(seq,y_test,y_score,file_name,auroc):
    folder_name = 'y_scores_output'
    if not os.path.exists(folder_name): # create folder for auc < 0.5
        os.makedirs(folder_name)
    basepath = os.getcwd()
    path = os.path.join(basepath,folder_name,file_name) # path for spesific ml
    data_information = (seq,y_test,y_score,auroc)  # form a tuple
    if os.path.exists(path): 
        auc_table = pd.read_csv(path)
        new_row_index = len(auc_table)
        auc_table.loc[new_row_index] = [*data_information]
        
    else: 
        auc_table = create_low_auc_table()
        auc_table.loc[0] = [*data_information]
    auc_table.to_csv(path,index=False)
    
def create_low_auc_table():
    columns = ["Seq","y_test","y_predict","auroc"]
    auc_table = pd.DataFrame(columns=columns)
    return auc_table







    



'''function check running time for amount of data points'''
def time_test(x_train,y_train):
    points = [100,1000,10000,100000] # Amount of data points to check
    for n in points:
        X_train_subset, y_train_subset = balance_data(x_train,y_train,n) # Balance data hopefully with n//2 for each label
        clf = get_classifier()
        start_time = time.time()
        clf.fit(X_train_subset,y_train_subset)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training {ML_TYPE} with {n} data points took {training_time:.4f} seconds.")

'''function to balance amount of y labels- e.a same amount of 1,0
agiasnt x features.
data_points - amount of wanted data points'''
def balance_data(x_train,y_train,data_points) -> tuple:
    # Get the indices of ones
    indices_ones = np.where(y_train == 1)[0]
    # Get the indices of zeros
    indices_zeros = np.where(y_train == 0)[0]
    # Try to split positive label and negative equaly
    if len(indices_ones) >= (data_points//2): # Enough positive equal amount
        n = (data_points//2) 
        m = n
    else: # Not enough positive, set positive to all positive amount and the rest negative
        n = len(indices_ones)
        m = data_points - n
    if m > len(indices_zeros): # Not enough negative points
        print(f"There are less data points then: {data_points}\nPlease change amount")
        exit(0)
        
    # Randomly pick n/2 indices from both ones and zeros
    random_indices_ones = np.random.choice(indices_ones, n, replace=False)
    random_indices_zeros = np.random.choice(indices_zeros, m, replace=False)
    # Merge indices
    combined_indices = np.concatenate((random_indices_ones, random_indices_zeros))
    y_train = y_train[combined_indices] 
    x_train = x_train[combined_indices]
    return (x_train,y_train) 


def get_sampler(balanced_ratio):
    sampler_type = input("1. over sampeling\n2. synthetic sampling\n")
    if sampler_type == "1": # over sampling
        return RandomOverSampler(sampling_strategy=balanced_ratio, random_state=42)
    else : return SMOTE(sampling_strategy=balanced_ratio,random_state=42)

def if_over_sample():
    global IF_OS 
    if_os = input("press y/Y to oversample, any other for more\n")
    if if_os.lower() == "y":
        sampler = get_sampler('auto')
        IF_OS = True
        
        return sampler
    else : return None
def get_sampler_type(sampler):
    if isinstance(sampler, RandomOverSampler):
        return "ROS"
    elif isinstance(sampler, SMOTE):
        return "SMOTE"
    else:
        return ""

def create_path_list(combined_folder):
    path_list = []
    for combined_file in os.listdir(combined_folder):
        combined_path = os.path.join(combined_folder,combined_file)
        path_list.append(combined_path)
    return path_list  
'''''' 
def create_feature_list(features_column):
    # Create a dictionary to store groups based on endings
    groups = {}

    # Group strings based on their endings
    for feature in features_column:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups

def run_only_seq(params,common_variables_ins):
    # 1. set variables:
    global ONLY_SEQ_INFO,IF_BP
    ONLY_SEQ_INFO = True
    IF_BP = False
    # 2. get data
    
    boolean_dict = {"if_bp" : False, "only_seq" : True}
    common_variables_ins.set_feature_booleans(boolean_dict)
    crisprsql(*params)
def run_with_epigenetic_features(params):
    global ONLY_SEQ_INFO,IF_BP,FEATURES_COLUMNS
    ONLY_SEQ_INFO = IF_BP = False
    # run corresponding features
    features_dict = create_feature_list(FEATURES_COLUMNS) # dict of feature by type (score, binary, fold)
    for feature_group in features_dict.values():
        for feature in feature_group: # run single feature
            FEATURES_COLUMNS = [feature]
            crisprsql(*params)
        if len(feature_group) == 1: # group containing one feature already been run with previous for loop
            continue
        FEATURES_COLUMNS = feature_group
        crisprsql(*params)
def run_with_bp_represenation(params,manager):
    global ONLY_SEQ_INFO,IF_BP,BP_PRESENTATION,ENCODED_LENGTH
    ONLY_SEQ_INFO = False
    IF_BP = True
    bw_copy = manager.get_bigwig_files() # gets a copy of the list
    for bw in bw_copy: # run each epi mark by file separtly
        manager.set_bigwig_files([bw])
        BP_PRESENTATION = 6 + manager.get_number_of_bigiwig() # should be 1
        ENCODED_LENGTH = 23 * BP_PRESENTATION
        crisprsql(*params)
    # run all epigentics mark togther
    manager.set_bigwig_files(bw_copy)
    BP_PRESENTATION = 6 + manager.get_number_of_bigiwig() # should be >= 1
    ENCODED_LENGTH = 23 * BP_PRESENTATION
    crisprsql(*params)
    # no need to close files will be closed by manager
def run_with_epi_spacial(params,manager):
    global ONLY_SEQ_INFO,IF_BP,IF_SEPERATE_EPI,EPIGENETIC_WINDOW_SIZE
    ONLY_SEQ_INFO = IF_BP = False
    IF_SEPERATE_EPI = True
    EPIGENETIC_WINDOW_SIZE = 2000
    bw_copy = manager.get_bigwig_files() # gets a copy of the list
    for bw in bw_copy: # run each epi mark by file separtly
        manager.set_bigwig_files([bw])
        crisprsql(*params)
    # run all epigentics mark togther
    # manager.set_bigwig_files(bw_copy)
    
    # crisprsql(*params)
       
def run_manualy(params,management):
    answer = input("press:\n1. only seq\n2. epigenetic features\n3. bp presentation\n4. epi seperate\n")
    if answer == "1":
        run_only_seq(params)
    elif answer == "2":
        run_with_epigenetic_features(params)
    elif answer == "3":
        run_with_bp_represenation(params,management)
    elif answer == "4":
        run_with_epi_spacial(params,management)
    else: 
        print("no good option, exiting.")
        exit(0)
'''function runs automation of all epigenetics combinations, onyl seq, and bp epigeneitcs represantion.'''
def auto_run(params,management,common_variables_ins):
    run_only_seq(params,common_variables_ins)
    run_with_epigenetic_features(params)
    run_with_bp_represenation(params,management)
    run_with_epi_spacial(params,management)

def run(params):
    global ML_TYPE
    ML_TYPE = input("Please enter ML type: (LOGREG, SVM, XGBOOST, XGBOOST_CW, RANDOMFOREST, CNN):\n") # ask for ML model
    management = File_management("pos","neg","/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/Chromstate","/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/bigwig")
    common_variables_ins = common_variables()
    sampler = if_over_sample() 
    
    new_params = params + (management, sampler,common_variables_ins)
    answer = input("press:\n1. auto\n2. manual\n")
    if answer == "1":
        auto_run(new_params,management,common_variables_ins)
    elif answer == "2":
        run_manualy(new_params,management,common_variables_ins)
    else:
        print("no good option, exiting.")
        exit(0)


if __name__ == "__main__":
     run((sys.argv[1],"change_csgs_globmax"))
     
    
