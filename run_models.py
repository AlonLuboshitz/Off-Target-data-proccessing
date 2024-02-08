# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
# "Chromstate_atacseq_peaks_score","Chromstate_atacseq_peaks_fold_enrichemnt","Chromstate_h3k4me3_peaks_score","Chromstate_h3k4me3_peaks_fold_enrichemnt"

FORCE_USE_CPU = False


#from Server_constants import BED_FILES_FOLDER, BIG_WIG_FOLDER, CHANGESEQ_GS_EPI
from constants import BED_FILES_FOLDER, BIG_WIG_FOLDER, MERGED_TEST, MERGED_CSGS_EPI 
from file_management import File_management
from features_engineering import generate_features_and_labels, order_data, get_tp_tn, extract_features
from evaluation import get_auc_by_tpr, get_tpr_by_n_expriments, evaluate_model
from models import get_cnn, get_logreg, get_xgboost, get_xgboost_cw

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE

import pandas as pd
import numpy as np
import sys
import time
import logging
import os
if FORCE_USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"  
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



class run_models:
    def __init__(self, file_manager) -> None:
        self.ml_type = "" # String inputed by user
        self.ml_name = ""
        self.ml_task = "" # class/reg
        self.shuffle = True
        self.if_os = False
        self.bp_presntation = 6
        self.guide_length = 23
        self.encoded_length =  self.guide_length * self.bp_presntation
        self.init_booleans()
        self.init_deep_hyper_params()
        self.epigenetic_window_size = 0
        if file_manager: # Not None
            self.file_manager = file_manager
        
        self.features_columns = ["Chromstate_atacseq_peaks_binary","Chromstate_h3k4me3_peaks_binary"]
    ## initairs
    def init_booleans(self):
        self.if_only_seq = self.if_seperate_epi = self.if_bp = self.if_epi_features = False  
    def init_deep_hyper_params(self):
        self.hyper_params = {'epochs': 5, 'batch_size': 1024, 'verbose' : 2}# Add any other fit parameters you need
    


    ## Features booleans setters
    def set_only_seq_booleans(self):
        self.if_bp = False
        self.if_only_seq = True
    
    def set_bp_in_seq_booleans(self):
        self.if_only_seq = False
        self.if_bp = True
        
    def set_epi_window_booleans(self):
        self.if_only_seq = self.if_bp = self.if_epi_features= False
        self.if_seperate_epi = True
    
    def set_epigenetic_feature_booleans(self):
        self.if_only_seq = self.if_bp = self.if_seperate_epi = False
        self.if_epi_features = True
    ## Model setters
    def set_hyper_params_class_wieghting(self, y_train):
        class_weights = compute_class_weight(class_weight='balanced',classes= np.unique(y_train),y= y_train)
        class_weight_dict = dict(enumerate(class_weights))
        self.hyper_params['class_weight'] = class_weight_dict

    def set_cross_validation(self):
        answer = input("Set cross validation method:\n1. Leave one out\n2. K cross validation\n")
        if answer == "1":
            self.cross_validation_method = "Leave_one_out"
        elif answer == "2":
            self.cross_validation_method = "K_cross"
            self.k = int(input("Set K (int): "))
    def set_model_name(self):
        answer = input("Please enter ML type: (LOGREG, XGBOOST, XGBOOST_CW, CNN):\n") # ask for ML model
        self.ml_name = answer
    ## Over sampling setter
    def set_over_sampling(self):
        if_os = input("press y/Y to oversample, any other for more\n")
        if if_os.lower() == "y":
            self.sampler = self.get_sampler('auto')
            self.if_os = True
        else: 
            self.sampler_type = ""
            self.sampler = None
    '''Tp are minority class, set the inverase ratio for xgb_cw
        args are 5 element tuple from get_tp_tn()'''
    def set_inverase_ratio(self, tps_tns):
        tprr, tp_test, tn_test, tp_train, tn_train = tps_tns # unpack tuple
        self.inverse_ratio = tn_train / tp_train

    ## Output setters
    '''1. File description based on booleans.
    Create a feature description list'''
    def set_features_output_description(self):
        if self.if_only_seq: # only seq
            self.features_description  = ["Only_Seq"]
        elif self.if_bp: # with base pair to gRNA bases or epigenetic window
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
        elif self.if_seperate_epi: # window size epenetics
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
            self.features_description.append(f'window_{self.epigenetic_window_size}')
        else : self.features_description = self.features_columns.copy() # features are added separtley
    '''2. Create file output name with .csv from:
    self: task - reg\clas, ml- cnn,rnn,etc.., sampler - ros,syntetic, features, ending - type of data'''
    def set_file_output_name(self, ending):
        self.set_features_output_description()
        # create feature f1_f2_f3...
        feature_str = "_".join(self.features_description)
        self.file_name = f'{self.ml_task}_{self.ml_name}_{self.cross_validation_method}_{self.sampler_type}_{feature_str}_{ending}.csv'

    '''3. Write results to output table:
    includes: ml type, auroc, auprc, unpacks 4 element tuple - tp,tn test, tp,tn train.
    features included for training the model
    what file/gRNA was left out.'''
    def write_to_table(self,auroc,auprc,file_left_out,table,Tpn_tuple,n_rank):
        try:
            new_row_index = len(table)  # Get the index for the new row
            table.loc[new_row_index] = [self.ml_name, auroc, auprc,*n_rank,*Tpn_tuple, self.features_description, file_left_out]  # Add data to the new row
        except: # empty data frame
            table.loc[0] = [self.ml_name , auroc, auprc,*Tpn_tuple , self.features_description , file_left_out]
        return table
    
    ### need to pass this function to file manager  
    '''write score vs test if auc<0.5'''
    def write_scores(self, seq,y_test,y_score,auroc):
        folder_name = 'y_scores_output'
        if not os.path.exists(folder_name): # create folder for auc < 0.5
            os.makedirs(folder_name)
        basepath = os.getcwd()
        path = os.path.join(basepath,folder_name,self.file_name) # path for spesific ml
        data_information = (seq,y_test,y_score,auroc)  # form a tuple
        if os.path.exists(path): 
            auc_table = pd.read_csv(path)
            new_row_index = len(auc_table)
            auc_table.loc[new_row_index] = [*data_information]
            
        else: 
            auc_table = self.create_low_auc_table()
            auc_table.loc[0] = [*data_information]
        auc_table.to_csv(path,index=False)
        
    def create_low_auc_table(self):
        columns = ["Seq","y_test","y_predict","auroc"]
        auc_table = pd.DataFrame(columns=columns)
        return auc_table
        
    ## Assistant RUN FUNCTIONS: DATA, MODEL, REGRESSION, CLASSIFICATION, 
        
    ## DATA:
    '''1. CREATE FEATURES VIA feature_engineering.py
    Use generate_features_and_lables from feature_engineering.py
    returns x_features, y_features, guide set'''
    def get_features(self): 
        return  generate_features_and_labels(self.file_manager.get_merged_data_path() , self.file_manager, self.encoded_length, self.bp_presntation, 
                                                                        self.if_bp, self.if_only_seq,self.if_seperate_epi,
                                                                        self.epigenetic_window_size,self.features_columns)
    # OVER SAMPLING:
    '''1. Get the sampler instace with ratio and set the sampler string'''
    def get_sampler(self,balanced_ratio):
        sampler_type = input("1. over sampeling\n2. synthetic sampling\n")
        if sampler_type == "1": # over sampling
            self.sampler_type = "ROS"
            return RandomOverSampler(sampling_strategy=balanced_ratio, random_state=42)
        else : 
            self.sampler_type = "SMOTE"
            return SMOTE(sampling_strategy=balanced_ratio,random_state=42)

    
    ## MODELS:
    '''1. GET MODEL'''
    def get_model(self):
        if self.ml_name == "LOGREG":
            self.ml_type = "ML"
            return get_logreg()
        elif self.ml_name == "XGBOOST":
            self.ml_type = "ML"
            return get_xgboost()
        elif self.ml_name == "XGBOOST_CW":
            self.ml_type = "ML"
            return get_xgboost_cw(self.inverse_ratio)
        elif self.ml_name == "CNN":
            self.ml_type = "DEEP"
            return get_cnn(self.guide_length, self.bp_presntation, self.if_only_seq, self.if_bp, 
                           self.if_seperate_epi, len(self.features_columns), self.epigenetic_window_size, self.file_manager.get_number_of_bigiwig())

    '''2. PREDICT'''
    def predict_with_model(self,X_train, y_train, X_test, y_test): # to run timetest unfold "#"
        # time_test(X_train,y_train)
        # exit(0)
        # train
        classifier = self.get_model()
        if self.ml_type == "DEEP":
            self.set_hyper_params_class_wieghting(y_train= y_train)
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)): 
                # if seperate epi/ only_seq=bp=false --> features added to seq encoding
                # extract featuers/epi window from sequence enconding 
                X_train = extract_features(X_train, self.encoded_length)
                X_test = extract_features(X_test,self.encoded_length)
            
            classifier.fit(X_train,y_train,**self.hyper_params)
            y_pos_scores_probs = classifier.predict(X_test,verbose = 2)
        else :
            classifier.fit(X_train,y_train)
            y_scores_probs = classifier.predict_proba(X_test)
            y_pos_scores_probs = y_scores_probs[:,1] # probalities for positive label (1 column for positive)
        return y_pos_scores_probs
    

    ## RUNNERS:
    # LEAVE OUT OUT
    def leave_one_out(self):
        # Set File output name
        self.set_file_output_name(self.out_put_name)
        # Set result table 
        results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
        # Get data
        x_features, y_labels, guides = self.get_features()
        print("Starting Leave-One-Out")
        for i,gRNA in enumerate(guides):
            # split data to i and ~ i
            x_train,y_train,x_test,y_test = order_data(x_features,y_labels,i+1,if_shuffle=True,if_print=False,sampler=self.sampler,if_over_sample=self.if_os)
            # get tps, tns and set inverse ratio
            self.pipe_line_model(x_train= x_train, y_train= y_train, x_test= x_test, y_test= y_test,
                                 iterations= len(guides), i_iter= i, key= gRNA, results_table= results_table)        
        
        results_table = results_table.sort_values(by="File_out")
        results_table.to_csv(self.file_name)
    
    ## K-CROSS VALIDATION
    '''Given k value split the data into K groups where the amount of intreset value is similar
    i.e. interest - positive amount, buldges amount, mismatch amount'''
    def k_cross_validation(self):
        # get data each x_feature has correspoding y_label
        x_features, y_labels, guides = self.get_features()
        # sum the y_labels > 0 for each array i in y_labels
        sum_labels = [np.sum(array > 0) for array in y_labels]        # sort the sum labels in Desc and get the sorted indices
        sorted_indices = np.argsort(sum_labels)[::-1]
        # init K groups and fill them with features by the sorted indices
        k_groups = self.fill_k_groups_indices(self.k, sum_labels = sum_labels, sorted_indices = sorted_indices)
        # Set File output name
        self.set_file_output_name(self.out_put_name)
        # Set result table 
        results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
        print("Starting K cross validation")
        for i in range(self.k): # iterate over the k groups, split to test/train by indices in I group
            x_train, y_train, x_test, y_test, test_guides = self.split_by_group(x_features, y_labels, k_groups, i, guides) 
            self.pipe_line_model(x_train, y_train, x_test, y_test, self.k, i+1 , test_guides, results_table)
        results_table = results_table.sort_values(by="File_out")
        results_table.to_csv(self.file_name)

    
    '''3. Split to x_test y_test by indices given in the k_group, rest of indices will be the training set'''
    def split_by_group(self,x_features, y_labels, k_groups, i, guides):
        # split to test indices and train indices
        test_indices = np.array(k_groups[i])
        # Concatenate arrays excluding k_groups[i]
        train_indices = np.concatenate(k_groups[:i] + k_groups[i+1:])
        x_test,y_test,x_train,y_train, test_guides = [], [], [], [], [] # init arrays for data type
        guides = list(guides)
        # Iterate on indices for test/train and fill arrays
        for idx in test_indices:
            x_test.append(x_features[idx])
            y_test.append(y_labels[idx])
            test_guides.append(guides[idx]) # Fill guides by test indices
        for idx in train_indices:
            x_train.append(x_features[idx])
            y_train.append(y_labels[idx])
        # Concatenate into np array and flatten the y arrays
        x_test = np.concatenate(x_test, axis= 0)
        x_train = np.concatenate(x_train, axis= 0)
        y_test = np.concatenate(y_test, axis= 0).ravel()
        y_train = np.concatenate(y_train, axis= 0).ravel()
        return x_train, y_train, x_test, y_test, test_guides

    '''2. Greedy approch to fill k groups with ~ equal amount of labels
        Getting sum of labels for each indice and sorted indices by sum filling the groups
        from the biggest amount to smallest adding to the minimum group'''
    def fill_k_groups_indices(self, k, sum_labels, sorted_indices):
        # Create k groups with 1 indice each from the sorted indices in descending order
        groups = [[sorted_indices[i]] for i in range(k)]
        
        
        for index in sorted_indices[k:]: # Notice [K:] to itreate over the remaining indices
        # Find the group with the smallest current sum
            min_sum_group = min(groups, key=lambda group: sum(sum_labels[i] for i in group), default=groups[0])
            # Add the series to the group with the smallest current sum
            min_sum_group.append(index)
        return groups
    
    
    
    def pipe_line_model(self, x_train, y_train, x_test, y_test, iterations, i_iter, key, results_table):
    # get tps, tns and set inverse ratio
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        self.set_inverase_ratio(tps_tns)
        # predict and evaluate model
        y_scores_probs = self.predict_with_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        auroc,auprc = evaluate_model(y_test = y_test, y_pos_scores_probs = y_scores_probs)
        n_rank_score = get_auc_by_tpr(tpr_arr=get_tpr_by_n_expriments(predicted_vals = y_scores_probs, y_test = y_test, n = 1000))
        print(f"Ith: {i_iter}\{iterations} split is done")
        # write scores
        results_table = self.write_to_table(auroc=auroc,auprc=auprc,file_left_out=key,table=results_table,Tpn_tuple=tps_tns,n_rank=n_rank_score)
        if auroc <= 0.5:
            self.write_scores(key,y_test,y_scores_probs,auroc)  
    
    ## RUN TYPES:
    # only seq
        
    def run_only_seq(self):
        # 1. set booleans:
        self.set_only_seq_booleans()
        self.run_by_validation_type()

        
    def run_with_epigenetic_features(self):
        # 1. set booleans
        self.set_epigenetic_feature_booleans()
        # run for each feature by its own and by group of features
        features_dict = split_epigenetic_features_into_groups(self.features_columns) # dict of feature by type (score, binary, fold)
        for feature_group in features_dict.values():
            for feature in feature_group: # run single feature
                self.features_columns = [feature]
                self.run_by_validation_type()
            if len(feature_group) == 1: # group containing one feature already been run with previous for loop
                continue
            self.features_columns = feature_group
            self.run_by_validation_type()
        
    def run_with_bp_represenation(self):
        self.set_bp_in_seq_booleans()
        bw_copy = self.file_manager.get_bigwig_files() # gets a copy of the list
        for bw in bw_copy: # run each epi mark by file separtly
            self.file_manager.set_bigwig_files([bw])
            self.bp_presntation = 6 +  self.file_manager.get_number_of_bigiwig() # should be 1
            self.encoded_length = 23 * self.bp_presntation
            self.run_by_validation_type()
        # run all epigentics mark togther
        self.file_manager.set_bigwig_files(bw_copy)
        self.bp_presntation = 6 +  self.file_manager.get_number_of_bigiwig() # should be 1
        self.encoded_length = 23 * self.bp_presntation
        self.run_by_validation_type()
        # no need to close files will be closed by manager

        
    def run_with_epi_spacial(self):
        # 1. set booleans
        self.set_epi_window_booleans()
        self.epigenetic_window_size = 2000
        bw_copy = self.file_manager.get_bigwig_files() # gets a copy of the list
        for bw in bw_copy: # run each epi mark by file separtly
            self.file_manager.set_bigwig_files([bw])
            self.run_by_validation_type()
        # run all epigentics mark togther
        self.file_manager.set_bigwig_files(bw_copy)
        self.run_by_validation_type()
    def run_by_validation_type(self):
        validation_functions = {
            'Leave_one_out': self.leave_one_out,
            'K_cross': self.k_cross_validation,
        }
        validation_function = validation_functions.get(self.cross_validation_method)
        if validation_function:
            validation_function()
        else:
            print("Invalid cross-validation method")
        
    def run_manualy(self):
        answer = input("press:\n1. only seq\n2. epigenetic features\n3. bp presentation\n4. epi seperate\n")
        if answer == "1":
            self.run_only_seq()
        elif answer == "2":
            self.run_with_epigenetic_features()
        elif answer == "3":
            self.run_with_bp_represenation()
        elif answer == "4":
            self.run_with_epi_spacial()
        else: 
            print("no good option, exiting.")
            exit(0)
    '''function runs automation of all epigenetics combinations, onyl seq, and bp epigeneitcs represantion.'''
    def auto_run(self):
        self.run_only_seq()
        self.run_with_epigenetic_features()
        self.run_with_bp_represenation()
        self.run_with_epi_spacial()

    def run(self, output_name):
        # set model (xgb, cnn, rnn)
        self.set_model_name()
        # file manager is on already
        # set sampler
        self.set_over_sampling()
        # set method
        self.set_cross_validation()
        self.out_put_name = output_name
        answer = input("press:\n1. auto\n2. manual\n")
        if answer == "1":
            self.auto_run()
        elif answer == "2":
            self.run_manualy()
        else:
            print("no good option, exiting.")
            exit(0)


## Epigenetic features helper
def split_epigenetic_features_into_groups(features_columns):
    # Create a dictionary to store groups based on endings
    groups = {}
    # Group strings based on their endings
    for feature in features_columns:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups
    



'''function check running time for amount of data points'''
def time_test(x_train,y_train):
    points = [100,1000,10000,100000] # Amount of data points to check
    for n in points:
        X_train_subset, y_train_subset = balance_data(x_train,y_train,n) # Balance data hopefully with n//2 for each label
        # clf = get_classifier()
        # start_time = time.time()
        # clf.fit(X_train_subset,y_train_subset)
        # end_time = time.time()
        # training_time = end_time - start_time
        # print(f"Training {ML_TYPE} with {n} data points took {training_time:.4f} seconds.")

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






if __name__ == "__main__":
    file_manger = File_management("","",BED_FILES_FOLDER,BIG_WIG_FOLDER, MERGED_TEST)
    runner_models = run_models(file_manager = file_manger) 
    runner_models.run("Test")
     
    