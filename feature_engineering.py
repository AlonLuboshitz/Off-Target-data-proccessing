# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
FEATURES_COLUMNS = ["grna_target_sequence"]
ONLY_SEQ_INFO = True #set as needed, if only seq then True.
LABEL = ["Label_negative"]
ML_TYPE = "" # String inputed by user
ENCODED_LENGTH = 6 * 23
SHUFFLE = True
import pandas as pd
import numpy as np
import sys
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
import os

#NOTE: if path for files is changed for more features, FEATURES_COLUMNS need to be changed.

'''paths for combined files containing target seq information and ot and labels.
create features and corresponding labels.
run logreg with leaving one file out for testing the data.
update results and extract csv file.
'''
def run_leave_one_out(guideseq40,guideseq50):
    file_paths = create_path_list(guideseq40) #+ create_path_list(guideseq50)    
    x_feature,y_label = generate_feature_labels(file_paths) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
    # leave one out - run model
    print("Starting ml")
    for i,path in enumerate(file_paths):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i)
       # run model
        #x_train,y_train = balance_data(x_train,y_train,12000)      
        auroc,auprc = get_ml_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        print(f"Ith: {i+1}\{len(file_paths)} split is done")
        ith_file_name = os.path.basename(path).split("_")[0]
        results_table = write_to_table(auroc=auroc,auprc=auprc,file_left_out=ith_file_name,table=results_table,ML_type=ML_TYPE,Tpn_tuple=tps_tns)
    
    file_name = create_file_name("")
    results_table = results_table.sort_values(by="File_out")
    results_table.to_csv(file_name)
'''function write to table by columsn:
ML type, auroc, auprc from log_reg, unpacks 4 element tuple - tp,tn test, tp,tn train.
features included for training the model
what file was left out.'''
def write_to_table(auroc,auprc,file_left_out,table,ML_type,Tpn_tuple):
    global FEATURES_COLUMNS
    if ONLY_SEQ_INFO:
        FEATURES_COLUMNS = ["Only_Seq"]
    try:
        new_row_index = len(table)  # Get the index for the new row
        table.loc[new_row_index] = [ML_type, auroc, auprc,*Tpn_tuple, FEATURES_COLUMNS, file_left_out]  # Add data to the new row
    except: # empty data frame
        table.loc[0] = [ML_type, auroc, auprc,*Tpn_tuple , FEATURES_COLUMNS, file_left_out]
    return table
def create_file_name(ending):
    global FEATURES_COLUMNS
    if ONLY_SEQ_INFO:
        FEATURES_COLUMNS = ["Only_Seq"]
    feature_str = ""
    for feature in FEATURES_COLUMNS:
      feature_str = feature_str + "_" + feature
    if SHUFFLE:
        shuffle_str = "with_shuffle_"
    else: shuffle_str = "no_shuflle_"
    file_name = ML_TYPE + "_" +  feature_str + "_" + shuffle_str + ending + ".csv"
    return file_name
def crisprsql(df_sql):
    df_sql = pd.read_csv(df_sql)
    # Set the threshold and update the column
    # threshold = 1e-5
    # print('cleavage>treshold: ',sum(df_sql['cleavage_freq']>1e-5))
    # print('cleavage<treshold: ',sum(df_sql['cleavage_freq']<=1e-5))
    # df_sql['cleavage_freq'] = df_sql['cleavage_freq'].apply(lambda x: 1 if x > threshold else 0)
    # print(sum(df_sql['cleavage_freq']==1))
    # print('measured=1: ',sum(df_sql['measured']==1))
    # print('measured=0: ',sum(df_sql['measured']==0))
    # before = sum(df_sql['measured']==1)
    # print(f"data length before target chr: {len(df_sql)}")
    df_sql = df_sql[df_sql['target_chr'] != '0']
    # print(f"data length after target chr: {len(df_sql)}")

    # after = sum(df_sql['measured']==1)
    # print(before-after)

    df_sql = df_sql[['target_sequence','grna_target_sequence','measured']]
    guides = set(df_sql['grna_target_sequence'])

    # Create a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames
    df_dict = {grna: group for grna, group in df_sql.groupby('grna_target_sequence')}

    # Create separate DataFrames for each gRNA in the set
    result_dataframes = {grna: df_dict.get(grna, pd.DataFrame()) for grna in guides}
    x_feature,y_label = generate_features_labels_sql(result_dataframes) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
    print("staring ml")
    file_name = create_file_name("crisprSql")
    for i,key in enumerate(result_dataframes.keys()):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i,if_shuffle=SHUFFLE,if_print=False)
        auroc,auprc,y_score = get_ml_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        print(f"Ith: {i+1}\{len(result_dataframes)} split is done")
        results_table = write_to_table(auroc=auroc,auprc=auprc,file_left_out=key,table=results_table,ML_type=ML_TYPE,Tpn_tuple=tps_tns)
        if auroc <= 0.5:
            write_scores(key,y_test,y_score,file_name,auroc)            
    results_table = results_table.sort_values(by="File_out")
    results_table.to_csv(file_name)
    # leave one out - run model
    pass
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



def generate_features_labels_sql(splited_guide_data):
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    for val in splited_guide_data.values():
        x_data = get_features(val,only_seq_info=ONLY_SEQ_INFO)
        x_data_all.append(x_data)
        y_labels_all.append(val[['measured']].values)
    return (x_data_all,y_labels_all)




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

def get_tp_tn(y_test,y_train):
    tp_train = np.count_nonzero(y_train) # count 1's
    tn_train = y_train.size - tp_train # count 0's
    if not tn_train == np.count_nonzero(y_train==0):
        print("error")
    tp_test = np.count_nonzero(y_test) # count 1's
    tn_test = y_test.size - tp_test #count 0's
    if not tn_test == np.count_nonzero(y_test==0):
        print("error")
    return (tp_test,tn_test,tp_train,tn_train)

'''funcion to run logsitic regression model and return roc,prc'''
def get_ml_auroc_auprc(X_train, y_train, X_test, y_test): # to run timetest unfold "#"
    # time_test(X_train,y_train)
    # exit(0)
    # train
    classifier = get_classifier()
    classifier.fit(X_train, y_train)
    # predict probs
    y_scores_probs = classifier.predict_proba(X_test)
    y_pos_scores_probs = y_scores_probs[:,1] # probalities for positive label (1 column for positive)
    # Calculate AUROC,AUPRC
    fpr, tpr, tresholds = roc_curve(y_test, y_pos_scores_probs)
    auroc = auc(fpr, tpr)
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pos_scores_probs)
    print("ML DONE")
    return (auroc,auprc,y_pos_scores_probs)

def get_classifier():
    if ML_TYPE == "LOGREG":
        return LogisticRegression(random_state=42,n_jobs=-1)
    elif ML_TYPE == "SVM":
        return SVC(kernel="linear",random_state=42)
    elif ML_TYPE == "RANDOMFOREST":
        return RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=1000)
def enforce_seq_length(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long
'''get x_axis features for ml algo.
data - data frame for guiderna
only_seq_info - bolean for only seq or other features.'''
def get_features(data, only_seq_info):
    x = data[FEATURES_COLUMNS].values
    
    seq_info = np.ones((x.shape[0], ENCODED_LENGTH))
    # siteseq, negative target seq
    for index, (otseq, grnaseq) in enumerate(zip(data['target_sequence'], data['grna_target_sequence'])):
        otseq = enforce_seq_length(otseq, 23)
        grnaseq = enforce_seq_length(grnaseq, 23)
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
    
'''given feature list, label list split them into
test and train data.
transform into ndarray and shuffle the train data'''
def order_data(X_feature,Y_labels,i,if_shuffle,if_print):
    # into nd array
    x_test = np.array(X_feature[i])
    y_test = np.array(Y_labels[i]).ravel() # flatten to one dimension the y label
    if if_print:
        varify_by_printing(X_feature,Y_labels,x_test,y_test,i,False,None)
    # exclude ith data from train set using slicing.
    x_train = X_feature[:i] + X_feature[i+1:]
    y_train = Y_labels[:i] + Y_labels[i+1:]
    # transform into ndarray
    x_train = np.concatenate(x_train,axis=0)
    y_train = np.concatenate(y_train,axis=0).ravel() # flatten to one dimension the y label
    if if_print:
        if i == len(X_feature) - 1:
            varify_by_printing(X_feature,Y_labels,x_train,y_train,i-2,False,None)
        else : varify_by_printing(X_feature,Y_labels,x_train,y_train,i-1,False,None)
    # shuffle the data keeping matching labels for corresponding x values.
    # permutation_indices = np.random.permutation(x_train.shape[0])
    # x_train = x_train[permutation_indices]
    # y_train = y_train[permutation_indices]
    # diffrenet shuffle
    # Generate or select random indices
    if if_shuffle:
        if if_print:
            indices_to_print = np.random.choice(len(x_train), 5, replace=False)
            print("Before Shuffling:") # Print examples before shuffling
            varify_by_printing(None,None,x_train,y_train,None,True,indices_to_print)
        # Shuffle the data
            x_train, y_train = shuffle(x_train, y_train, random_state=42)
            print("\nAfter Shuffling:")  # Print examples after shuffling
            varify_by_printing(None,None,x_train,y_train,None,True,indices_to_print)
        else:   
            x_train, y_train = shuffle(x_train, y_train, random_state=42)

    return (x_train,y_train,x_test,y_test)
def varify_by_printing(X_feature,Y_labels,x_sub,y_sub,i,if_shuffle,indices):
    if not if_shuffle:
        for j in range(5):
            print("X_feature_val: ",X_feature[i][j])
            print("x_sub_val: ",x_sub[j])
            print("Y_labels_val: ",Y_labels[i][j])
            print("y_sub_val: ",y_sub[j])
    else: 
        for idx in indices:
            print(f"Index {idx}: Input: {x_sub[idx]}, Target: {y_sub[idx]}")

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
''' need to add over/under sampling'''       
def over_under_sample(X_train,y_train) -> tuple:
    num_minority_samples_before = sum(y_train == 1)
    # Apply RandomOverSampler to the training data
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    # Count the number of samples in the minority class after oversampling
    num_minority_samples_after = sum(y_train == 1)
    # Calculate the number of samples that have been duplicated
    num_samples_duplicated = num_minority_samples_after - num_minority_samples_before
    print(f"Number of samples duplicated: {num_samples_duplicated}")
    return (X_train,y_train)
def set_if_seq():
    if_seq = input("press y/Y to keep only_seq, any other for more\n")
    if not if_seq.lower() == "y":
        return False
    else: return True
def create_path_list(combined_folder):
    path_list = []
    for combined_file in os.listdir(combined_folder):
        combined_path = os.path.join(combined_folder,combined_file)
        path_list.append(combined_path)
    return path_list   
if __name__ == "__main__":
    ONLY_SEQ_INFO = set_if_seq()
    ML_TYPE = input("Please enter ML type: (LOGREG, SVM, XGBOOST, RANDOMFOREST):\n")
    #run_leave_one_out(sys.argv[1],sys.argv[2])
    crisprsql(sys.argv[1])
    
