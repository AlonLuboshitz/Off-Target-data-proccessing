# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
# "Chromstate_atacseq_peaks_score","Chromstate_atacseq_peaks_fold_enrichemnt","Chromstate_h3k4me3_peaks_score","Chromstate_h3k4me3_peaks_fold_enrichemnt"

FORCE_CPU = True
from features_engineering import  order_data, get_tp_tn, extract_features, get_guides_indexes
from evaluation import get_auc_by_tpr, get_tpr_by_n_expriments, evaluate_classification_model, evaluate_model
from models import get_cnn, get_logreg, get_xgboost, get_xgboost_cw, get_gru_emd, argmax_layer
from utilities import validate_dictionary_input, get_memory_usage
from parsing import features_method_dict, cross_val_dict, model_dict, class_weights_dict
from features_and_model_utilities import get_encoding_parameters, split_epigenetic_features_into_groups
from train_and_test_utilities import split_to_train_and_val, split_by_guides
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
import numpy as np
import time
import os
import signal
import tensorflow as tf


def tf_clean_up():
    tf.keras.backend.clear_session()
    print("GPU memory cleared.")
def set_signlas_clean_up():
    signal.signal(signal.SIGINT, tf_clean_up)
    signal.signal(signal.SIGSTOP, tf_clean_up)

class run_models:
    def __init__(self) -> None:
        self.ml_type = self.ml_name = self.ml_task = None
        
        self.shuffle = True
        self.if_os = False
        self.os_valid = False
        self.init_encoded_params = False
        self.init_booleans()
        self.init_model_dict()
        self.init_cross_val_dict()
        self.init_features_methods_dict()
        self.set_computation_power(FORCE_CPU)
        self.epigenetic_window_size = 0
        self.features_columns = "" 

    ## initairs ###
    # This functions are used to init necceseray parameters in order to run a model.
    # If the parameters are not setted before running the model, the program will raise an error.

    def set_computation_power(self, force_cpu=False):
        '''
        This function checks if there is an available GPU for computation.
        If GPUs are available, it enables memory growth. 
        If force_cpu is True, it forces the usage of CPU instead of GPU.
        '''
        if force_cpu:
            # Forcing CPU usage by hiding GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.gpu_available = False
            print("Forcing CPU computation, no GPU will be used.")
            return
        gpus = tf.config.list_physical_devices('GPU')  # Stable API for listing GPUs
        if gpus:
            try:
                # Enabling memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')  # Logical devices are virtual GPUs
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
                gpu_available = True  # Indicating that GPU is available
                #set_signlas_clean_up()
            except RuntimeError as e:
                # Handle the error if memory growth cannot be set
                print(f"Error enabling memory growth: {e}")
                gpu_available = False
        else:
            gpu_available = False
            print("No GPU found. Using CPU.")
        self.gpu_available = gpu_available
    def init_booleans(self):
        '''Features booleans'''
        self.if_only_seq = self.if_seperate_epi = self.if_bp = self.if_features_by_columns = False  
        self.method_init = False
    

    def init_deep_parameters(self, epochs = 5, batch_size = 1024, verbose = 2):
        '''Deep learning hyper parameters'''
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.hyper_params = {'epochs': self.epochs, 'batch_size': self.batch_size, 'verbose' : self.verbose, 'callbacks' : [], 'validation_data': None}# Add any other fit parameters you need
    
    def init_early_stoping(self):
        '''Early stopping for deep learning models'''
        
        if self.early:
            if self.paitence > 0:
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.paitence, restore_best_weights=True)
            else : early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.epochs, restore_best_weights=True)
            
        self.hyper_params.setdefault('callbacks', []).append(early_stopping)
   


    def init_model_dict(self):
        ''' Create a dictionary for ML models'''
        self.model_dict = model_dict()
        self.model_type_initiaded = False
    
    def init_cross_val_dict(self):
        ''' Create a dictionary for cross validation methods'''
        self.cross_val_dict = cross_val_dict()
        self.cross_val_init = False
    
    def init_features_methods_dict(self):
        ''' Create a dictionary for running methods'''
        self.features_methods_dict = features_method_dict()
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
        elif not self.ml_task:
            raise RuntimeError("ML task - classification/regression was not set")
        elif not self.init_encoded_params:
            raise RuntimeError("Encoded parameters were not set")
        

    def setup_runner(self, ml_task = None, model_num = None, cross_val = None, features_method = None, 
                     over_sampling = None, cw = None, encoding_type = None, if_bulges = None , early_stopping = None , deep_parameteres = None):
        '''
        This function sets the parameters for the model.
        Args:
        1. ml_task - classification or regression (str)
        2. model_num - number of the model to use (int)
        3. cross_val - cross validation method # Maybe can remove (int)
        4. features_method - what feature included in the model (int)
        5. over_sampling - if to use over sampling/downsampling (str)
        6. cw - class wieghting - 1- true, 2 -false (int)
        7. encoding_type - encoding type for the model and features (int)
        8. if_bulges - if to include bulges in the encoding (Bool)
        9. early_stopping - if to use early stopping - Tuple (Bool, number of epochs)
        10. deep_parameteres - Tuple of epochs, batch size, verbose
        '''
        self.set_model_task(ml_task)
        self.set_model(model_num, deep_parameteres)
        self.set_cross_validation(cross_val)
        self.set_features_method(features_method)
        self.set_over_sampling('n') # set over sampling
        self.set_class_wieghting(cw)
        self.set_early_stopping(early_stopping[0],early_stopping[1])
        self.set_encoding_parameters(encoding_type, if_bulges)
        self.set_data_reproducibility(False) # set data, model reproducibility
        self.set_model_reproducibility(False)
        self.set_functions_dict()
        self.validate_initiation()
        self.init = True
    
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
    
    def set_class_wieghting(self, cw):
        answer = validate_dictionary_input(cw, class_weights_dict())
        if answer == 1:
            self.cw = True
        else : self.cw = False
    
    def set_early_stopping(self, if_early = 1,paitence = None):
        self.early = False
        self.paitence = 0
        if if_early == 1:
            self.early = True
            if paitence > 0:
                self.paitence = int(paitence)
            self.init_early_stoping()
    
    def set_encoding_parameters(self,enconding_type, if_bulges):
        self.guide_length, self.bp_presntation = get_encoding_parameters(enconding_type, if_bulges)
        self.encoded_length =  self.guide_length * self.bp_presntation
        if self.ml_name == "GRU-EMB":
            self.bp_presntation = self.bp_presntation**2
            self.encoded_length =  self.guide_length * self.bp_presntation
        self.init_encoded_params = True
    '''Set seeds for reproducibility'''
    def set_deep_seeds(self,seed=42):
        self.seed = seed
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
        self.seed = loc_seed
    ## Features booleans setters
    def set_only_seq_booleans(self):
        self.if_bp = False
        self.if_only_seq = True
    
    def set_bp_in_seq_booleans(self):
        self.if_only_seq = False
        self.if_bp = True
        
    def set_epi_window_booleans(self):
        self.if_only_seq = self.if_bp = self.if_features_by_columns= False
        self.if_seperate_epi = True
    
    def set_features_by_columns_booleans(self):
        self.if_only_seq = self.if_bp = self.if_seperate_epi = False
        self.if_features_by_columns = True

    def get_model_booleans(self):
        '''Return the booleans for the model
        ----------
        Tuple of - only_seq, bp, seperate_epi, epi_features, data_reproducibility, model_reproducibility'''
        return self.if_only_seq, self.if_bp, self.if_seperate_epi, self.if_features_by_columns, self.data_reproducibility, self.model_reproducibility
   
    ## Model setters ###
    # This functions are used to set the model parameters.
    def set_hyper_params_class_wieghting(self, y_train):
        if self.ml_task == "Classification":
            if self.cw:
                class_weights = compute_class_weight(class_weight='balanced',classes= np.unique(y_train),y= y_train)
                class_weight_dict = dict(enumerate(class_weights))
                self.hyper_params['class_weight'] = class_weight_dict
        else :  return # no class wieghting for regression
    
    def set_model(self, model_num_answer = None , deep_parameters = None):
        if self.model_type_initiaded:
            return # model was already set
        if not self.ml_task:
            raise RuntimeError("Model task need to be setted before the model")
        model_num_answer = validate_dictionary_input(model_num_answer, self.model_dict)
        '''Given an answer - model number, set the ml_type and ml name'''
    
        if model_num_answer < 4: # ML models
            self.ml_type = "ML"
        else : # Deep models
            self.ml_type = "DEEP"
            self.init_deep_parameters(*deep_parameters if deep_parameters else None)
        self.ml_name = self.model_dict[model_num_answer]
        self.model_type_initiaded = True
            
    
    def set_model_task(self, task):
        '''This function set the model Task - classification or regression'''
        if self.ml_task:
            return # task was already set
        if task.lower() == "classification":
            self.ml_task = "Classification"
        elif task.lower() == "regression" or task.lower() == "t_regression":
            self.ml_task = "Regression"
        else : raise ValueError("Task must be classification or regression/t_regression")

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
            # self.k = int(input("Set K (int): "))
        elif cross_val_answer == 3:
            self.cross_validation_method = "Ensemble"
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
            2: self.set_features_by_columns_booleans,
            3: self.set_bp_in_seq_booleans,
            4: self.set_epi_window_booleans
            
        }   
        booleans_dict[feature_method_answer]()
        self.feature_type = self.features_methods_dict[feature_method_answer]
        self.method_init = True

    '''Set features columns for the model'''
    def set_features_columns(self, features_columns):
        if features_columns:
            self.features_columns = features_columns
        else: raise RuntimeError("Trying to set features columns for model where no features columns were given")

    def set_big_wig_number(self, number):
        if isinstance(number,int) and number >= 0:
            self.bigwig_numer = number 
        else : raise ValueError("Number of bigwig files must be a non negative integer")
    def get_parameters_by_names(self):
        '''This function returns the following atributes by their names:
        Ml_name, Cross_validation_method, Features_type, epochs, batch_size'''
        return self.ml_name, self.cross_validation_method, self.feature_type, [self.epochs,self.batch_size], self.early
    def get_gpu_availability(self):
        return self.gpu_available
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

    ## Output setters: Features used, file name, model evalution and results table ##

    ## 1. File description based on booleans.
   
    def set_features_output_description(self):
        '''Create a feature description list'''
        if self.if_only_seq: # only seq
            self.features_description  = ["Only-Seq"]
        elif self.if_bp: # with base pair to gRNA bases or epigenetic window
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
        elif self.if_seperate_epi: # window size epenetics
            self.features_description = [file_name[0] for file_name in self.file_manager.get_bigwig_files()]
            self.features_description.append(f'window_{self.epigenetic_window_size}')
        else : self.features_description = self.features_columns.copy() # features are added separtley
    
    ## 2. Create file output name with .csv from:
    # self: task - reg\clas, ml- cnn,rnn,etc.., sampler - ros,syntetic, features, data set name
    def set_file_output_name(self, data_set_name):
        self.set_features_output_description()
        # create feature f1_f2_f3...
        feature_str = "_".join(self.features_description)
        self.file_name = f'{self.ml_task}_{self.ml_name}_{self.k}{self.cross_validation_method}_{self.sampler_type}_{feature_str}_{data_set_name}'

    ## 3. Set output table
    def set_output_table(self):
        if self.ml_task == "Classification":
            results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
        elif self.ml_task == "Regression":
            results_table = pd.DataFrame(columns=['ML_type', 'R_pearson','P.pearson','R_spearman','P.spearman','MSE','OTSs','N', 'Features', 'File_out'])
        return results_table
    ## 4. Write results to output table:
    def write_to_table(self,auroc,auprc,file_left_out,table,Tpn_tuple,n_rank):
        '''This function writes the results to the results table.
    includes: ml type, auroc, auprc, unpacks 4 element tuple - tp,tn test, tp,tn train.
    features included for training the model
    what file/gRNA was left out.'''
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
    '''1. GET MODEL - from models.py'''
    def get_model(self):
        if self.ml_name == "LOGREG":
            return get_logreg(self.random_state, self.data_reproducibility)
        elif self.ml_name == "XGBOOST":
            return get_xgboost(self.random_state)
        elif self.ml_name == "XGBOOST_CW":
            return get_xgboost_cw(self.inverse_ratio, self.random_state,self.data_reproducibility)
        elif self.ml_name == "CNN":
            return get_cnn(self.guide_length, self.bp_presntation, self.if_only_seq, self.if_bp, 
                           self.if_seperate_epi, len(self.features_columns), self.epigenetic_window_size, self.bigwig_numer, self.ml_task)
        elif self.ml_name == "RNN":
            pass
        elif self.ml_name == "GRU-EMB":
            return get_gru_emd(task=self.ml_task,input_shape=(self.guide_length,self.bp_presntation),num_of_additional_features=len(self.features_columns),if_flatten=True)
    '''2. Training and Predicting with model:'''
    ## Train model: if Deep learning set class wieghting and extract features
    def train_model(self,X_train, y_train):
        if not self.init:
            raise RuntimeError("Trying to trian a model without a setup - please re run the code and use setup_runner function")
        # time_test(X_train,y_train)
        model = self.get_model()
        if self.ml_type == "DEEP":
            self.set_hyper_params_class_wieghting(y_train= y_train)
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)): 
                # if seperate epi/ only_seq=bp=false --> features added to seq encoding
                # extract featuers/epi window from sequence enconding 
                X_train = extract_features(X_train, self.encoded_length)
            if self.early: # split to train and val
                X_train,y_train,x_y_val = split_to_train_and_val(X_train,y_train,self.ml_task, seed=self.seed)
            self.hyper_params["validation_data"] = x_y_val
            model.fit(X_train,y_train,**self.hyper_params)
        else :
            model.fit(X_train,y_train)
        print(f"Memory Usage train model: {get_memory_usage():.2f} MB")
        tf.keras.backend.clear_session()
        return model
    
    def predict_with_model(self, model, X_test):
        if not self.init:
            raise RuntimeError("Trying to predict with a model without a setup - please re run the code and use setup_runner function")
        if self.ml_type == "DEEP":
            if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)):
                X_test = extract_features(X_test,self.encoded_length)
            y_pos_scores_probs = model.predict(X_test,verbose = 2,batch_size=self.hyper_params['batch_size'])
        else :
            y_scores_probs = model.predict_proba(X_test)
            y_pos_scores_probs = y_scores_probs[:,1]
        return y_pos_scores_probs

    ## RUNNERS: LEAVE ONE OUT, K-CROSS VALIDATION, ENSEMBLE
    # LEAVE OUT OUT
    def leave_one_out(self, x_features = None, y_labels = None, guides = None,  ):
        # Set File output name
        self.set_file_output_name(self.out_put_name)
        # Set result table 
        results_table = self.set_output_table()
        # Get data
        x_features, y_labels, guides = self.get_features()
        print("Starting Leave-One-Out")
        for i,gRNA in enumerate(guides):
            # split data to i and ~ i
            x_train,y_train,x_test,y_test = order_data(x_features,y_labels,i+1,if_shuffle=True,if_print=False,sampler=self.sampler,if_over_sample=self.if_os)
            # get tps, tns and set inverse ratio
            self.train_and_evaluate_model(x_train= x_train, y_train= y_train, x_test= x_test, y_test= y_test,
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
            self.train_and_evaluate_model(x_train, y_train, x_test, y_test, self.k, i+1 , test_guides, results_table)
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
    
    
    
    def train_and_evaluate_model(self, x_train, y_train, x_test, y_test, iterations, i_iter, key, results_table = None):
        '''Run the model for each split. 
        If results table is given write the results to the table.
        If no table is given return the score made by the model. 
        '''
        # get tps, tns and set inverse ratio
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        self.set_inverase_ratio(tps_tns)
        self.set_model_reproducibility(self.model_reproducibility)
        # predict and evaluate model
        model = self.train_model(X_train=x_train,y_train=y_train)
        y_scores_probs = self.predict_with_model(model = model, X_test = x_test)
        if results_table is None:
            print(f"Ith: {i_iter}\{iterations} split is done")
            return y_scores_probs
        else: 
            evaluate_model(y_test = y_test, y_pos_scores_probs = y_scores_probs)
        auroc,auprc = evaluate_classification_model(y_test = y_test, y_pos_scores_probs = y_scores_probs)
        n_rank_score = get_auc_by_tpr(tpr_arr=get_tpr_by_n_expriments(predicted_vals = y_scores_probs, y_test = y_test, n = 1000))
        print(f"Ith: {i_iter}\{iterations} split is done")
        # write scores
        results_table = self.write_to_table(auroc=auroc,auprc=auprc,file_left_out=key,table=results_table,Tpn_tuple=tps_tns,n_rank=n_rank_score)
        if auroc <= 0.5:
            self.write_scores(key,y_test,y_scores_probs,auroc)  
    
    ## ENSEMBLE:
    def create_ensemble(self, n_models, output_path, guides_train_list, seed_addition = 0, x_features=None, y_labels=None,guides=None):
        '''This function create ensemble of n_models and save them in the output path.
        The models train on the guide list given in guides_train_list.
        Each model created with diffrenet intitaion seed + a seed addition. This can be usefull to reproduce the model.
        Positive ratio is the ratio of positive labels in the training set, if None all the positive labels will be used.
        Args:
        1. n_models - number of models to create.
        2. output_path - path to save the models.
        3. guides_train_list - list of guides to train on.
        4. seed_addition - int to add to the seed for reproducibility.
        5. positive_ratio - list of ratios for positive labels in the training set.
        if positive ratio given, for each ratio a new folder will be created in the output path.
        6. X_train, y_train - if given will be used for training the models.
        ----------
        Saves: n trained models in output_path.
        Example: create_ensebmle(5,"/models",["ATT...TGG",...],seed_addition=10,positive_ratio=[0.5,0.7,0.9])'''
        
        for j in range(n_models):
            temp_path = os.path.join(output_path,f"model_{j+1}.keras")
            print(f'Creating model {j+1} out of {n_models}')
            self.create_model(output_path=temp_path,guides_train_list=guides_train_list,seed_addition=(j+1+seed_addition),x_features=x_features,y_labels=y_labels,guides=guides)

    
    def create_model(self, output_path, guides_train_list, seed_addition = 10, x_features=None, y_labels=None,guides=None):
        if x_features is None or y_labels is None or guides is None:
            raise RuntimeError("Cannot create ensemble without data : x_features, y_labels, guides")
        else: 
            x_train,y_train,g_idx = split_by_guides(guides, guides_train_list, x_features, y_labels)
        self.set_deep_seeds(seed = seed_addition) # repro but random init (j+1 not 0)
        model = self.train_model(X_train=x_train,y_train=y_train)
        model.save(output_path)
    def test_ensmbel(self, ensembel_model_list, tested_guide_list,x_features=None, y_labels=None,guides=None):
        '''This function tests the models in the given ensmble.
        By defualt it test the models on the tested_guide_list, If test_on_guides is False:
        it will test on the guides that are not in the tested_guide_list
        Args:
        1. ensembel_model_list - list of paths to the models
        2. tested_guide_list - list of guides to test on
        3. test_on_guides - boolean to test on the given guides or on the diffrence guides'''
        # Get data
        if x_features is None or y_labels is None or guides is None:
            raise RuntimeError("Cannot test ensemble without data : x_features, y_labels, guides")
        x_test, y_test, guides_idx = split_by_guides(guides, tested_guide_list, x_features, y_labels)
        all_guides_idx = get_guides_indexes(guide_idxs=guides_idx) # get indexes of all grna,ots
        # init 2d array for y_scores 
        # Row - model, Column - probalities
        y_scores_probs = np.zeros(shape=(len(ensembel_model_list), len(y_test))) 
        for index,model_path in enumerate(ensembel_model_list): # iterate on models and predict y_scores
            model = tf.keras.models.load_model(model_path, custom_objects={'argmax_layer': argmax_layer})
            # self.set_random_seeds(seed = (index+1+additional_seed))
            model_predictions = self.predict_with_model(model=model,X_test=x_test).ravel() # predict and flatten to 1d
            y_scores_probs[index] = model_predictions
        return y_scores_probs, y_test, all_guides_idx
    
    def test_model(self, model_path, tested_guide_list,x_features=None, y_labels=None,guides=None):
        x_test, y_test, guides_idx = split_by_guides(guides, tested_guide_list, x_features, y_labels)
        all_guides_idx = get_guides_indexes(guide_idxs=guides_idx) # get indexes of all grna,ots
        #model = tf.keras.models.load_model(model_path, safe_mode=False)

        model = tf.keras.models.load_model(model_path, custom_objects={'argmax_layer': argmax_layer})
        model_predictions = self.predict_with_model(model=model,X_test=x_test).ravel() # predict and flatten to 1d
        return model_predictions, y_test, all_guides_idx
    ## RUN TYPES:
    # only seq
        
    def run_only_seq(self):
        # 1. set booleans:
        self.set_only_seq_booleans()
        self.run_by_validation_type()

    '''Run with epigenetic features, split to groups and run each group and all together.'''    
    def run_with_epigenetic_features(self):
        # 1. set booleans
        self.set_features_by_columns_booleans()
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





     

