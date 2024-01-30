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

from features_engineering import  extract_features

import pandas as pd
import numpy as np
import time

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, average_precision_score
import logging
import os
if FORCE_USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"  
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from tensorflow import keras
from keras.layers import Reshape, Conv1D, Input, Dense, Flatten, Concatenate, MaxPooling1D, Reshape, Dropout

'''write score vs test if auc<0.5'''
def write_scores(self,seq,y_test,y_score,file_name,auroc):
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
        auc_table = self.create_low_auc_table()
        auc_table.loc[0] = [*data_information]
    auc_table.to_csv(path,index=False)
    
def create_low_auc_table(self):
    columns = ["Seq","y_test","y_predict","auroc"]
    auc_table = pd.DataFrame(columns=columns)
    return auc_table







def get_tp_tn(y_test,y_train):
    tp_train = np.count_nonzero(y_train) # count 1's
    tn_train = y_train.size - tp_train # count 0's
    if not tn_train == np.count_nonzero(y_train==0):
        print("error")
    tp_test = np.count_nonzero(y_test) # count 1's
    tn_test = y_test.size - tp_test #count 0's
    if not tn_test == np.count_nonzero(y_test==0):
        print("error")
    tp_ratio = tp_test / (tp_test + tn_test)
    return (tp_ratio,tp_test,tn_test,tp_train,tn_train)

'''funcion to run logsitic regression model and return roc,prc'''
def get_ml_auroc_auprc(self,X_train, y_train, X_test, y_test): # to run timetest unfold "#"
    # time_test(X_train,y_train)
    # exit(0)
    # train
    tprr,tpt,tnt,m,n = get_tp_tn(y_test=y_test,y_train=y_train)
    inverse_ratio = tnt / tpt
    class_weights = compute_class_weight(class_weight='balanced',classes= np.unique(y_train),y= y_train)
    class_weight_dict = dict(enumerate(class_weights))
    FIT_PARAMS['class_weight'] = class_weight_dict
    classifier = self.get_classifier(ratio=inverse_ratio)
    if isinstance(classifier,keras.Model):

        if self.if_seperate_epi or (not (self.if_only_seq or self.if_bp)): #only-seq = bp = false -> only features
            # bp True != only-seq
            # seperate
            X_train_seq,X_train_features = extract_features(X_train,ENCODED_LENGTH)
            X_train = [X_train_seq,X_train_features]
            X_test_seq,X_test_features = extract_features(X_test,ENCODED_LENGTH)
            X_test = [X_test_seq,X_test_features]
        
        classifier.fit(X_train,y_train,**FIT_PARAMS)
        y_pos_scores_probs = classifier.predict(X_test,verbose = 2)
    else :
        classifier.fit(X_train,y_train)
        y_scores_probs = classifier.predict_proba(X_test)
        y_pos_scores_probs = y_scores_probs[:,1] # probalities for positive label (1 column for positive)
    
    # # Calculate AUROC,AUPRC
    fpr, tpr, tresholds = roc_curve(y_test, y_pos_scores_probs)
    auroc = auc(fpr, tpr)
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pos_scores_probs)
    print("ML DONE")
    return (auroc,auprc,y_pos_scores_probs)

'''get the true positive rate for up to n expriemnets by calculating:
the first n prediction values, what the % of positive predcition out of the the TP amount.
calculate auc value for 1-n'''
def get_tpr_by_n_expriments(predicted_vals,y_test,n):
    # valid that test amount is more then n
    if n > len(y_test):
        print(f"n expriments: {n} is bigger then data points amount: {len(y_test)}, n set to data points")
        n = len(y_test)
    
    tp_amount = np.count_nonzero(y_test) # get tp amount
    if predicted_vals.ndim > 1:
        predicted_vals = predicted_vals.ravel()
    sorted_indices = np.argsort(predicted_vals)[::-1] # Get the indices that would sort the prediction values array in descending order    
    tp_amount_by_prediction = 0 # set tp amount by prediction
    tpr_array = np.empty(0) # array for tpr
    for i in range(n):
        # y_test has label of 1\0 if 1 adds it to tp_amount
        tp_amount_by_prediction = tp_amount_by_prediction + y_test[sorted_indices[i]]
        tp_rate = tp_amount_by_prediction / tp_amount
        tpr_array= np.append(tpr_array,tp_rate)
        if tp_rate == 1.0:
            # tp amount == tp amount in prediction no need more expriments and all tp are found
            tpr_array = np.concatenate((tpr_array, np.ones(n - (i + 1)))) # fill the tpr array with 1 
            break    
    return tpr_array  

def get_auc_by_tpr(tpr_arr):
    amount_of_points = len(tpr_arr)
    x_values = np.arange(1, amount_of_points + 1) # x values by lenght of tpr_array
    calculated_auc = auc(x_values,tpr_arr)
    calculated_auc = calculated_auc / amount_of_points # normalizied auc
    return calculated_auc,amount_of_points
    

def get_classifier(ratio):
    if ML_TYPE == "LOGREG":
        return LogisticRegression(random_state=42,n_jobs=-1)
    elif ML_TYPE == "SVM":
        return SVC(kernel="linear",random_state=42)
    elif ML_TYPE == "RANDOMFOREST":
        return RandomForestClassifier(random_state=42,n_jobs=-1)
    elif ML_TYPE == "XGBOOST":
        return XGBClassifier(random_state=42, objective='binary:logistic',n_jobs=-1)
    elif ML_TYPE == "XGBOOST_CW":
        return XGBClassifier(scale_pos_weight=ratio,random_state=42, objective='binary:logistic',n_jobs=-1)
    elif ML_TYPE == "CNN":
        return create_convolution_model(sequence_length= GUIDE_LENGTH,bp_presenation= BP_PRESENTATION,only_seq_info=ONLY_SEQ_INFO,if_bp=IF_BP,num_of_additional_features=len(FEATURES_COLUMNS),epigenetic_window_size=EPIGENETIC_WINDOW_SIZE)
    
def get_cnn(sequence_length, bp_presenation, only_seq_info, if_bp, num_of_additional_features, epigenetic_window_size):
        return create_convolution_model(sequence_length, bp_presenation, only_seq_info, if_bp, num_of_additional_features, epigenetic_window_size)   
def get_xgboost_cw(scale_pos_weight):
        return XGBClassifier(scale_pos_weight=scale_pos_weight,random_state=42, objective='binary:logistic',n_jobs=-1) 
def get_xgboost():
        return XGBClassifier(random_state=42, objective='binary:logistic',n_jobs=-1)
def get_logreg():
        return LogisticRegression(random_state=42,n_jobs=-1)
def create_conv_seq_layers(self,seq_input,sequence_length,bp_presenation):
    seq_input_reshaped = Reshape((sequence_length, bp_presenation)) (seq_input)

    seq_conv_1 = Conv1D(32, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_input_reshaped)
    seq_acti_1 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_1)
    seq_drop_1 = keras.layers.Dropout(0.1)(seq_acti_1)
    
    seq_conv_2 = Conv1D(64, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_drop_1)
    seq_acti_2 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_2)
    seq_max_pooling_1 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_2)

    seq_conv_3 = Conv1D(128, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_1)
    seq_acti_3 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_3)

    seq_conv_4 = Conv1D(256, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_acti_3)
    seq_acti_4 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_4)
    seq_max_pooling_2 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_4)

    seq_conv_5 = Conv1D(512, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_2)
    seq_acti_5 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_5)

    seq_flatten = Flatten()(seq_acti_5) 
    return seq_flatten
def create_conv_epi_layer(self,epi_input,kernal_size,strides,epigenetic_window_size):
    epi_input_reshaped = Reshape((epigenetic_window_size,1))(epi_input)
    epi_conv_6 = Conv1D(2,kernel_size=kernal_size,kernel_initializer='random_uniform',strides=strides,padding='valid')(epi_input_reshaped)
    epi_acti_6 = keras.layers.LeakyReLU(alpha=0.2)(epi_conv_6)
    epi_max_pool_3 = MaxPooling1D(pool_size=2,strides=2, padding='same')(epi_acti_6) 
    epi_seq_flatten = Flatten()(epi_max_pool_3)
    return epi_seq_flatten

def create_convolution_model(self,sequence_length, bp_presenation,only_seq_info,if_bp,num_of_additional_features,epigenetic_window_size):
    # set seq conv layers
    seq_input = Input(shape=(sequence_length * bp_presenation))
    seq_flatten = self.create_conv_seq_layers(seq_input=seq_input,sequence_length=sequence_length,bp_presenation=bp_presenation)

    if (only_seq_info or if_bp): # only seq information given
        combined = seq_flatten
    elif IF_SEPERATE_EPI: # epigenetics in diffrenet conv
        epi_feature = Input(shape=(epigenetic_window_size))
        epi_seq_flatten = self.create_conv_epi_layer(epi_input=epi_feature,kernal_size=(int(epigenetic_window_size/10)),strides=5,epigenetic_window_size=epigenetic_window_size)
        combined = Concatenate()([seq_flatten, epi_seq_flatten])
        
    else:
        feature_input = Input(shape=(num_of_additional_features))
        combined = Concatenate()([seq_flatten, feature_input])

    seq_dense_1 = Dense(256, activation='relu')(combined)
    seq_drop_2 = keras.layers.Dropout(0.3)(seq_dense_1)
    seq_dense_2 = Dense(128, activation='relu')(seq_drop_2)
    seq_drop_3 = keras.layers.Dropout(0.2)(seq_dense_2)
    seq_dense_3 = Dense(64, activation='relu')(seq_drop_3)
    seq_drop_4 = keras.layers.Dropout(0.2)(seq_dense_3)
    seq_dense_4 = Dense(40, activation='relu')(seq_drop_4)
    seq_drop_5 = keras.layers.Dropout(0.2)(seq_dense_4)

    output = Dense(1, activation='sigmoid')(seq_drop_5)
    if (only_seq_info or if_bp):
        model = keras.Model(inputs=seq_input, outputs=output)
    elif IF_SEPERATE_EPI:
        model = keras.Model(inputs=[seq_input,epi_feature], outputs=output)

    else:
        model = keras.Model(inputs=[seq_input, feature_input], outputs=output)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['binary_accuracy'])
    print(model.summary())
    return model



'''function check running time for amount of data points'''
def time_test(self,x_train,y_train):
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
def balance_data(self, x_train,y_train,data_points) -> tuple:
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


def get_sampler(self,balanced_ratio):
    sampler_type = input("1. over sampeling\n2. synthetic sampling\n")
    if sampler_type == "1": # over sampling
        return RandomOverSampler(sampling_strategy=balanced_ratio, random_state=42)
    else : return SMOTE(sampling_strategy=balanced_ratio,random_state=42)

def if_over_sample(self):
    global IF_OS 
    if_os = input("press y/Y to oversample, any other for more\n")
    if if_os.lower() == "y":
        sampler = self.get_sampler('auto')
        IF_OS = True
        
        return sampler
    else : return None
def get_sampler_type(self, sampler):
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
def create_feature_list(self,features_column):
    # Create a dictionary to store groups based on endings
    groups = {}

    # Group strings based on their endings
    for feature in features_column:
        ending = feature.split("_")[-1]  # last part after _ "can be score, enrichment, etc.."
        groups.setdefault(ending, []).append(feature)
    return groups

    

