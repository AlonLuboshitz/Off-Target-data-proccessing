# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
# "Chromstate_atacseq_peaks_score","Chromstate_atacseq_peaks_fold_enrichemnt","Chromstate_h3k4me3_peaks_score","Chromstate_h3k4me3_peaks_fold_enrichemnt"

FORCE_USE_CPU = False

from features_engineering import generate_features_and_labels, order_data, get_tp_tn, extract_features, get_guides_indexes
from evaluation import get_auc_by_tpr, get_tpr_by_n_expriments, evaluate_model
from models import get_cnn, get_logreg, get_xgboost, get_xgboost_cw
from utilities import validate_dictionary_input, split_epigenetic_features_into_groups
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
import numpy as np
import time
import logging
import os
if FORCE_USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"  
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
        self.os_valid = False
        self.bp_presntation = 6
        self.guide_length = 23
        self.encoded_length =  self.guide_length * self.bp_presntation
        self.init_booleans()
        self.init_model_dict()
        self.init_cross_val_dict()
        self.init_features_methods_dict()
        self.epigenetic_window_size = 0
        if file_manager: # Not None
            self.file_manager = file_manager
        else : raise RuntimeError("Trying to init model runner without file manager")
        
        self.features_columns = ["Chromstate_atacseq_peaks_binary","Chromstate_h3k4me3_peaks_binary"]
    ## initairs
    def init_booleans(self):
        self.if_only_seq = self.if_seperate_epi = self.if_bp = self.if_epi_features = False  
        self.method_init = False
    
    def init_deep_hyper_params(self):
        self.hyper_params = {'epochs': 5, 'batch_size': 1024, 'verbose' : 2}# Add any other fit parameters you need
    
    def init_model_dict(self):
        ''' Create a dictionary for ML models'''
        self.model_dict = {
            1: "LOGREG",
            2: "XGBOOST",
            3: "XGBOOST_CW",
            4: "CNN",
            5: "RNN"
        }
        self.model_type_initiaded = False
    
    def init_cross_val_dict(self):
        ''' Create a dictionary for cross validation methods'''
        self.cross_val_dict = {
            1: "Leave_one_out",
            2: "K_cross",
            3: "Ensmbel"
        }
        self.cross_val_init = False
    
    def init_features_methods_dict(self):
        ''' Create a dictionary for running methods'''
        self.features_methods_dict = {
            1: "Only_sequence",
            2: "Epigenetics_by_features",
            3: "Base_pair_epigenetics_in_Sequence",
            4: "Spatial_epigenetics"
        }
        self.method_init = False
    
    def validate_initiation(self):
        if not self.model_type_initiaded:
            raise RuntimeError("Model type was not set")
        elif not self.method_init:
            raise RuntimeError("Method type was not set")
        elif not self.cross_val_init:
            raise RuntimeError("Cross validation type was not set")
        elif not self.os_valid:
            raise RuntimeError("Over sampling was not set")
        

    def setup_runner(self, model_num = None, cross_val = None, features_method = None, over_sampling = None):
        self.set_model(model_num)
        self.set_cross_validation(cross_val)
        self.set_features_method(features_method)
        self.set_over_sampling('n') # set over sampling
        self.set_data_reproducibility(False) # set data, model reproducibility
        self.set_model_reproducibility(False)
        self.set_functions_dict()
        self.validate_initiation()
        self.init = True
    
    # def setup_ensmbel_runner(self):
    #     self.set_model()
    #     self.set_boleans_method()
    #     self.set_over_sampling()
    #     self.set_data_reproducibility(False)
    #     self.set_model_reproducibility(False)
    #     self.init = True
    '''Set reproducibility for data and model'''
    def set_data_reproducibility(self, bool):
        self.data_reproducibility = bool
    def set_model_reproducibility(self, bool):
        self.model_reproducibility = bool
        if self.model_reproducibility:
            if self.ml_type == "DEEP":
                self.set_deep_seeds()
            else : 
                self.set_ml_seeds()
        else : # set randomness
            self.set_random_seeds(False)
    '''Set seeds for reproducibility'''
    def set_deep_seeds(self,seed=42):
        tf.random.set_seed(seed) # Set seed for Python's random module (used by TensorFlow internally)
        tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()
    def set_ml_seeds(self):
        #np.random.seed(42) # set np seed
        self.random_state = 42
    def set_random_seeds(self,seed):
        loc_seed =int(time.time())
        if seed:
            loc_seed = seed
        tf.random.set_seed(loc_seed) # Set seed for Python's random module (used by TensorFlow internally)
        tf.keras.utils.set_random_seed(loc_seed)  # sets seeds for base-python, numpy and tf

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
    
    def set_model(self, model_num_answer = None):
        if self.model_type_initiaded:
            return # model was already set
        model_num_answer = validate_dictionary_input(model_num_answer, self.model_dict)
        '''Given an answer - model number, set the ml_type and ml name'''
        if model_num_answer > 0 and model_num_answer < len(self.model_dict):
            if model_num_answer < 4: # ML models
                self.ml_type = "ML"
            else : # Deep models
                self.ml_type = "DEEP"
                self.init_deep_hyper_params()
            self.ml_name = self.model_dict[model_num_answer]
            self.model_type_initiaded = True
            self.file_manager.add_type_to_models_paths(self.ml_name) # add model name to models and results path

    def set_cross_validation(self, cross_val_answer = None):
        if not self.model_type_initiaded:
            raise RuntimeError("Model type need to be setted before cross val type")
        if self.cross_val_init:
            return # cross val was already set
        ''' Set cross validation method and k value if needed.'''
        cross_val_answer = validate_dictionary_input(cross_val_answer, self.cross_val_dict)
        if cross_val_answer == 1:
            self.cross_validation_method = "Leave_one_out"
            self.k = ""
        elif cross_val_answer == 2:
            self.cross_validation_method = "K_cross"
            self.k = int(input("Set K (int): "))
        elif cross_val_answer == 3:
            self.cross_validation_method = "Ensemble"
        self.file_manager.add_type_to_models_paths(self.cross_validation_method) # add cross_val to models and results path
        self.cross_val_init = True
    
    def set_features_method(self, feature_method_answer = None):  
        if not self.model_type_initiaded and not self.cross_val_init:
            raise RuntimeError("Model type and cross val need to be setted before features method")
        if self.method_init:
            return # method was already set
        '''Set running method'''
        feature_method_answer = validate_dictionary_input(feature_method_answer, self.features_methods_dict)
        booleans_dict = {
            1: self.set_only_seq_booleans,
            2: self.set_epigenetic_feature_booleans,
            3: self.set_bp_in_seq_booleans,
            4: self.set_epi_window_booleans
        }   
        booleans_dict[feature_method_answer]()
        self.file_manager.add_type_to_models_paths(self.features_methods_dict[feature_method_answer]) # add method to models and results path
        self.method_init = True

    '''Set features columns for the model'''
    def set_features_columns(self, features_columns):
        if features_columns:
            self.features_columns = features_columns
        else: raise RuntimeError("Trying to set features columns for model where no features columns were given")
        

               
    ## Over sampling setter
    def set_over_sampling(self, over_sampling):
        if self.os_valid:
            return # over sampling was already set
        if not over_sampling:
            if_os = input("press y/Y to oversample, any other for more\n")
        else : if_os = over_sampling
        if if_os.lower() == "y":
            self.sampler = self.get_sampler('auto')
            self.if_os = True
            
        else: 
            self.sampler_type = ""
            self.sampler = None
        self.os_valid = True
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
            self.features_description  = ["Only-Seq"]
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
        self.file_name = f'{self.ml_task}_{self.ml_name}_{self.k}{self.cross_validation_method}_{self.sampler_type}_{feature_str}_{ending}'

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
        self.set_random_seeds(False)
        return  generate_features_and_labels(self.file_manager.get_merged_data_path() , self.file_manager, self.encoded_length, self.bp_presntation, 
                                                                        self.if_bp, self.if_only_seq,self.if_seperate_epi,
                                                                        self.epigenetic_window_size,self.features_columns, self.data_reproducibility)
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
            return get_logreg(self.random_state, self.data_reproducibility)
        elif self.ml_name == "XGBOOST":
            return get_xgboost(self.random_state)
        elif self.ml_name == "XGBOOST_CW":
            return get_xgboost_cw(self.inverse_ratio, self.random_state,self.data_reproducibility)
        elif self.ml_name == "CNN":
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
    ## Train model: if Deep learning set class wieghting and extract features
    def train_model(self,X_train, y_train):
        classifier = self.get_model()
        if self.ml_type == "DEEP":
            self.set_hyper_params_class_wieghting(y_train= y_train)
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)): 
                # if seperate epi/ only_seq=bp=false --> features added to seq encoding
                # extract featuers/epi window from sequence enconding 
                X_train = extract_features(X_train, self.encoded_length)
            classifier.fit(X_train,y_train,**self.hyper_params)
        else :
            classifier.fit(X_train,y_train)
        return classifier
    ## Predict on classifier
    def predict_with_model(self, classifier, X_test):
        if self.ml_type == "DEEP":
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)):
                X_test = extract_features(X_test,self.encoded_length)
            y_pos_scores_probs = classifier.predict(X_test,verbose = 2,batch_size=self.hyper_params['batch_size'])
        else :
            y_scores_probs = classifier.predict_proba(X_test)
            y_pos_scores_probs = y_scores_probs[:,1]
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
        #keep_groups(k_groups, sum_labels, guides, self.k, f"Test_{self.k}K_guides.csv")
        
        # Set File output name
        self.set_file_output_name(self.out_put_name)
        # Set result table 
        results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
        print("Starting K cross validation")
        for i in range(self.k): # iterate over the k groups, split to test/train by indices in I group
            x_train, y_train, x_test, y_test, test_guides = self.split_by_group(x_features, y_labels, k_groups, i, guides) 
            self.pipe_line_model(x_train, y_train, x_test, y_test, self.k, i+1 , test_guides, results_table)
        results_table = results_table.sort_values(by="File_out")
        self.file_manager.save_ml_results(results_table, self.file_name)


    
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

    
    def fill_k_groups_indices(self, k, sum_labels, sorted_indices):
        '''2. Greedy approch to fill k groups with ~ equal amount of labels
        Getting sum of labels for each indice and sorted indices by sum filling the groups
        from the biggest amount to smallest adding to the minimum group'''
        # Create k groups with 1 indice each from the sorted indices in descending order
        if k < len(sorted_indices):
            raise RuntimeError("K value is smaller than the amount of guides")
        groups = [[sorted_indices[i]] for i in range(k)]
        
        
        for index in sorted_indices[k:]: # Notice [K:] to itreate over the remaining indices
        # Find the group with the smallest current sum
            min_sum_group = min(groups, key=lambda group: sum(sum_labels[i] for i in group), default=groups[0])
            # Add the series to the group with the smallest current sum
            min_sum_group.append(index)
        return groups
    
    
    '''Run the model for each split, write the results to the table and if auc < 0.5 write the scores to a file'''
    def pipe_line_model(self, x_train, y_train, x_test, y_test, iterations, i_iter, key, results_table):
    # get tps, tns and set inverse ratio
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        self.set_inverase_ratio(tps_tns)
        self.set_model_reproducibility(self.model_reproducibility)
        # predict and evaluate model
        classifier = self.train_model(X_train=x_train,y_train=y_train)
        
        y_scores_probs = self.predict_with_model(classifier=classifier,X_test=x_test)
       
        auroc,auprc = evaluate_model(y_test = y_test, y_pos_scores_probs = y_scores_probs)
        n_rank_score = get_auc_by_tpr(tpr_arr=get_tpr_by_n_expriments(predicted_vals = y_scores_probs, y_test = y_test, n = 1000))
        print(f"Ith: {i_iter}\{iterations} split is done")
        # write scores
        results_table = self.write_to_table(auroc=auroc,auprc=auprc,file_left_out=key,table=results_table,Tpn_tuple=tps_tns,n_rank=n_rank_score)
        if auroc <= 0.5:
            self.write_scores(key,y_test,y_scores_probs,auroc)  
    
    ## ENSEMBLE:
    def create_ensemble(self, n_models, output_path, guides_train_list, seed_addition = 0):
        if not self.init:
            raise RuntimeError("Trying to run model without setup")
        # Get data
        # self.set_data_reproducibility(True)
        x_features, y_labels, guides = self.get_features()
        guides_idx = self.keep_intersect_guides_indices(guides, guides_train_list) # keep only the train guides indexes
        if (len(guides_idx) == len(guides)): # All guides are for training
            x_train = np.concatenate(x_features, axis= 0)
            y_train = np.concatenate(y_labels, axis= 0).ravel()
        else:
            x_train, y_train = self.split_by_indexes(x_features, y_labels, guides_idx) # split by traing indexes
        for j in range(n_models):
            self.set_deep_seeds(seed = (j+1+seed_addition)) # repro but random init (j+1 not 0)
            classifier = self.train_model(X_train=x_train,y_train=y_train)
            temp_path = os.path.join(output_path,f"model_{j+1}.keras")
            classifier.save(temp_path)
    
    def test_ensmbel(self, ensembel_model_list, tested_guide_list,test_on_guides = True):
        '''This function tests the models in the given ensmble.
        By defualt it test the models on the tested_guide_list, If test_on_guides is False:
        it will test on the guides that are not in the tested_guide_list
        Args:
        1. ensembel_model_list - list of paths to the models
        2. tested_guide_list - list of guides to test on
        3. test_on_guides - boolean to test on the given guides or on the diffrence guides'''
        # Get data
        x_features, y_labels, guides = self.get_features()
        guides_idx = self.keep_intersect_guides_indices(guides, tested_guide_list) # keep only the test guides indexes
        all_guides_idx = get_guides_indexes(guide_idxs=guides_idx) # get indexes of all grna,ots
        x_test, y_test = self.split_by_indexes(x_features, y_labels, guides_idx) # split by test indexes
        # init 2d array for y_scores 
        # Row - model, Column - probalities
        y_scores_probs = np.zeros(shape=(len(ensembel_model_list), len(y_test))) 
        for index,model_path in enumerate(ensembel_model_list): # iterate on models and predict y_scores
            classifier = tf.keras.models.load_model(model_path)
            # self.set_random_seeds(seed = (index+1+additional_seed))
            model_predictions = self.predict_with_model(classifier=classifier,X_test=x_test).ravel() # predict and flatten to 1d
            y_scores_probs[index] = model_predictions
        return y_scores_probs, y_test, all_guides_idx
    def keep_diffrence_guides_indices(self, guides, test_guides):
        '''This function returns the indexes of the guides that are NOT in the given test_guides 
        Be good to train/test on the guides not presnted in the given guides 
        Args:
        1. guides - list of guides
        2. test_guides - list of guides to keep if exists in guides'''
        return [idx for idx, guide in enumerate(guides) if guide not in test_guides]
    def keep_intersect_guides_indices(self,guides, test_guides):
        '''This function returns the indexes of the guides that are in the given test_guides 
        Be good to train/test on the guides given 
        Args:
        1. guides - list of guides
        2. test_guides - list of guides to keep if exists in guides'''
        return [idx for idx, guide in enumerate(guides) if guide in test_guides]
    
    def split_by_indexes(self, x_features, y_labels, indices):
        x_, y_ = [], [] 
        
        for idx in indices:
            x_.append(x_features[idx])
            y_.append(y_labels[idx])
        # Concatenate into np array and flatten the y arrays
        x_ = np.concatenate(x_, axis= 0)
        y_ = np.concatenate(y_, axis= 0).ravel()
        return x_, y_

        

    ## RUN TYPES:
    # only seq
        
    def run_only_seq(self):
        # 1. set booleans:
        self.set_only_seq_booleans()
        self.run_by_validation_type()

    '''Run with epigenetic features, split to groups and run each group and all together.'''    
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
    '''Run with bp representation, run each epi mark by file separtly and all together.'''   
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

    '''Run with epi seperatly from sequence (only valid for Deep)
      run each epi mark by file separtly and all together.'''    
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
    def set_functions_dict(self):  
        self.functions_dict = {
            1: ("Only sequence",self.run_only_seq),
            2: ("Epigenetic by features",self.run_with_epigenetic_features),
            3: ("Base pair epigenetic in Sequence",self.run_with_bp_represenation),
            4: ("Seperate epigenetics ",self.run_with_epi_spacial)
        }   
    def run_manualy(self):
        
        # Choose a method to run by the dictionary
        print("Choose a method to run: ")
        for key, value in self.functions_dict.items():
            print(f"{key}: {value[0]}")
        answer = input()
        answer = int(answer)
        # Validate input by dictionary
        validate_dictionary_input(answer, self.functions_dict) # validate input by data method dictionary
        # If valid run the method (1 for second tuple element in the dictionary value tuple)
        self.functions_dict[int(answer)][1]()
       
    
    '''function runs automation of all epigenetics combinations, onyl seq, and bp epigeneitcs represantion.'''
    def auto_run(self, function_number):
        validate_dictionary_input(function_number, self.functions_dict) # validate input by data method dictionary
        self.functions_dict[function_number][1]()
        if self.ml_type == "ML" and function_number == 3: # machine learing runs bp+seq but not seperate epi
            self.run_with_bp_represenation()
        elif function_number == 4 :
            self.run_with_epi_spacial() # deep learning run seperate epi
    '''auto_run is a parameter to run the model automaticly or manualy
    its a dictionary with the following: key- run_values, values - tuple of boolean and number of the function to run.'''
    def run(self, auto_run_bool, function_number ,output_name):
        if not self.init:
            self.setup_runner()
        self.out_put_name = output_name
        if auto_run_bool:
            self.auto_run(function_number)
        else:
            self.run_manualy()
        print("Done")  


## Epigenetic features helper

    



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





     
    
