# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
FEATURES_COLUMNS = ["Mtyhlation_293_h3k4me3_ENCFF498ERO"]
ONLY_SEQ_INFO = False #set as needed, if only seq then True.
LABEL = ["Label_negative"]
ENCODED_LENGTH = 6 * 23
import pandas as pd
import numpy as np
import sys
from chromatin_labeling import create_path_list
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, average_precision_score
import os

#NOTE: if path for files is changed for more features, FEATURES_COLUMNS need to be changed.

'''paths for combined files containing target seq information and ot and labels.
create features and corresponding labels.
run logreg with leaving one file out for testing the data.
update results and extract csv file.
'''
def run_leave_one_out(guideseq40,guideseq50):
    file_paths = create_path_list(guideseq50) #+ create_path_list(guideseq50)    
    x_feature,y_label = generate_feature_labels(file_paths) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc', 'Features', 'File_out'])
    # leave one out - run model
    for i,path in enumerate(file_paths):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i)
        # print (f'x_train: {x_train[:15]}, y_train: {y_train[:15]}')
        # num_ones = np.count_nonzero(y_train)
        # run logistic model
        auroc,auprc = get_logreg_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        print(f"Ith: {i+1}\{len(file_paths)} split is done")
        ith_file_name = os.path.basename(path).split("_")[0]
        results_table = write_to_table(auroc=auroc,auprc=auprc,file_left_out=ith_file_name,table=results_table,ML_type="log_reg")
    results_table.to_csv('ML_results.csv')
'''function write to table: auroc,auprc from log_reg.
what file was left out.'''
def write_to_table(auroc,auprc,file_left_out,table,ML_type):
    try:
        new_row_index = len(table)  # Get the index for the new row
        table.loc[new_row_index] = [ML_type, auroc, auprc, FEATURES_COLUMNS, file_left_out]  # Add data to the new row
    except: # empty data frame
        table.loc[0] = [ML_type, auroc, auprc, FEATURES_COLUMNS, file_left_out]
    return table
    
    

'''given feature list, label list split them into
test and train data.
transform into ndarray and shuffle the train data'''
def order_data(X_feature,Y_labels,i):
    # into nd array
    x_test = np.array(X_feature[i])
    y_test = np.array(Y_labels[i]).ravel() # flatten to one dimension the y label
    # exclude ith data from train set using slicing.
    x_train = X_feature[:i] + X_feature[i+1:]
    y_train = Y_labels[:i] + Y_labels[i+1:]
    # transform into ndarray
    x_train = np.concatenate(x_train,axis=0)
    y_train = np.concatenate(y_train,axis=0).ravel() # flatten to one dimension the y label
    # shuffle the data keeping matching labels for corresponding x values.
    permutation_indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation_indices]
    y_train = y_train[permutation_indices]
    return (x_train,y_train,x_test,y_test)

'''function to balance amount of y labels- e.a same amount of 1,0
agiasnt x features.'''
def balance_data(y_train,x_train):
    # Get the indices of ones
    indices_ones = np.where(y_train == 1)[0]
    # Get the indices of zeros
    indices_zeros = np.where(y_train == 0)[0]
    # Randomly pick the same number of indices as the number of ones
    random_indices_zeros = np.random.choice(indices_zeros, len(indices_ones), replace=False)
    # Merge indices
    combined_indices = np.concatenate((indices_ones, random_indices_zeros))
    y_train = y_train[combined_indices] 
    x_train = x_train[combined_indices]
    return (y_train,x_train)
def generate_feature_labels(path_list):
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    for file_path in (path_list):
        # Load data from the file
        data = pd.read_csv(file_path)
        #print(data[FEATURES_COLUMNS]) check data
        # Get x_data using your get_features function
        x_data = get_features(data, only_seq_info=ONLY_SEQ_INFO)  # Set only_seq_info as needed
        
        # Append x_data to the list
        x_data_all.append(x_data)
        
        # Append y_labels to the list
        y_labels_all.append(data[LABEL].values)
    # return lists
    return (x_data_all,y_labels_all)
    
'''funcion to run logsitic regression model and return roc,prc'''
def get_logreg_auroc_auprc(X_train, y_train, X_test, y_test): #
    logreg_classifier = LogisticRegression(random_state=42, n_jobs=-1)
    logreg_classifier.fit(X_train, y_train)
    y_scores_logreg = logreg_classifier.predict(X_test)
    # Calculate AUROC
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_scores_logreg)
    auroc_logreg = auc(fpr_logreg, tpr_logreg)
    # Calculate AUPRC
    auprc_logreg = average_precision_score(y_test, y_scores_logreg)
    print("LOGREG DONE")
    return (auroc_logreg,auprc_logreg)

'''get x_axis features for ml algo.
data - data frame for guiderna
only_seq_info - bolean for only seq or other features.'''
def get_features(data, only_seq_info):
    x = data[FEATURES_COLUMNS].values
    
    seq_info = np.ones((x.shape[0], ENCODED_LENGTH))
    for index, (otseq, grnaseq) in enumerate(zip(data['Siteseq'], data['TargetSequence_negative'])):
        # sequence = enforce_seq_length(sequence, FORCED_LENGTH)
        # sequence2 = enforce_seq_length(sequence2, FORCED_LENGTH)
        otseq = otseq.upper()
        seq_info[index] = seq_to_one_hot(otseq, grnaseq)
    if only_seq_info:
        x = seq_info
    else:
        x = np.append(x, seq_info, axis = 1)
    return x
'''encoding for grna and off target
creating 6 dimnesional vector * length of sequence(23 bp)
first 4 dimnesions are for A,T,C,G
last 2 dimnesions are indicating which letter belongs to which sequence.
returned flatten vector'''
def seq_to_one_hot(sequence, seq_guide):
    bases = ['A', 'T', 'C', 'G']
    onehot = np.zeros(ENCODED_LENGTH, dtype=int)
    sequence_length = len(sequence)
    for i in range(sequence_length):
        for key, base in enumerate(bases):
            if sequence[i] == base:
                onehot[6 * i + key] = 1
            if seq_guide[i] == base:
                onehot[6 * i + key] = 1
        if sequence[i] != seq_guide[i]:  # Mismatch
            try:
                if bases.index(sequence[i]) < bases.index(seq_guide[i]):
                    onehot[6 * i + 4] = 1
                else:
                    onehot[6 * i + 5] = 1
            except ValueError:  # Non-ATCG base found
                pass
    return onehot

if __name__ == "__main__":
    run_leave_one_out(sys.argv[1],sys.argv[2])
    
