# python script for feature engineering - for each guideseq exp retrive:
# x - (Guide rna (TargetSeq),Off-target(Siteseq)) --> one hot enconding
# y - label (1 - active off target), (0 - inactive off target)
# ENCONDING : vector of 6th dimension represnting grna and offtarget sequences and missmatches.
FEATURES_COLUMNS = ["Chromstate_atacseq_peaks_score"]
ONLY_SEQ_INFO = True #set as needed, if only seq then True.
LABEL = ["Label"]
ML_TYPE = "" # String inputed by user
BP_PRESENTATION = 6
ENCODED_LENGTH =  23 * BP_PRESENTATION
SHUFFLE = True
TARGET = "target"
OFFTARGET = "offtarget_sequence"
CHROM = "chrom"
START = "chromStart"
END = "chromEnd"
IF_BP = False
BIGWIG = 0
from file_management import File_management
#AMOUNT_OF_BP_EPI = 1
import pandas as pd
import numpy as np
import sys
import time
import pyBigWig
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
    file_paths = create_path_list(guideseq40) + create_path_list(guideseq50)    
    x_feature,y_label = generate_feature_labels(file_paths) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
    file_name = create_file_name("lables")
    # leave one out - run model
    print("Starting ml")
    for i,path in enumerate(file_paths):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i,if_shuffle=SHUFFLE,if_print=False)
       # run model
        #x_train,y_train = balance_data(x_train,y_train,12000)      
        auroc,auprc,y_score = get_ml_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        print(f"Ith: {i+1}\{len(file_paths)} split is done")
        ith_file_name = os.path.basename(path).split("_")[0]
        results_table = write_to_table(auroc=auroc,auprc=auprc,file_left_out=ith_file_name,table=results_table,ML_type=ML_TYPE,Tpn_tuple=tps_tns)
        if auroc <= 0.5:
            write_scores(ith_file_name,y_test,y_score,file_name,auroc) 
    results_table = results_table.sort_values(by="File_out")
    results_table.to_csv(file_name)
'''function write to table by columsn:
ML type, auroc, auprc from log_reg, unpacks 4 element tuple - tp,tn test, tp,tn train.
features included for training the model
what file was left out.'''
def write_to_table(auroc,auprc,file_left_out,table,ML_type,Tpn_tuple,n_rank):
    global FEATURES_COLUMNS
    if ONLY_SEQ_INFO:
        FEATURES_COLUMNS = ["Only_Seq"]
    try:
        new_row_index = len(table)  # Get the index for the new row
        table.loc[new_row_index] = [ML_type, auroc, auprc,*n_rank,*Tpn_tuple, FEATURES_COLUMNS, file_left_out]  # Add data to the new row
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
def crisprsql(data_table,target_colmun,off_target_column,y_column,file_name,file_manager):
    data_table = pd.read_csv(data_table)
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
    #df_sql = df_sql[df_sql['target_chr'] != '0']
    # print(f"data length after target chr: {len(df_sql)}")

    # after = sum(df_sql['measured']==1)
    # print(before-after)
    #data_table = data_table[['target_sequence','grna_target_sequence','measured']]
    
    guides = set(data_table[target_colmun]) # set unquie guide identifier

    # Create a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames
    df_dict = {grna: group for grna, group in data_table.groupby(target_colmun)}

    # Create separate DataFrames for each gRNA in the set
    result_dataframes = {grna: df_dict.get(grna, pd.DataFrame()) for grna in guides}
    x_feature,y_label = generate_features_labels_sql(result_dataframes,target_column=target_colmun,off_target_column=off_target_column,y_column=y_column,manager=file_manager) # List of arrays
    results_table = pd.DataFrame(columns=['ML_type', 'Auroc', 'Auprc','N-rank','N','Tp-ratio','T.P_test','T.N_test','T.P_train','T.N_train', 'Features', 'File_out'])
    print("staring ml")
    file_name = create_file_name(file_name)
    for i,key in enumerate(result_dataframes.keys()):
        x_train,y_train,x_test,y_test = order_data(x_feature,y_label,i,if_shuffle=SHUFFLE,if_print=False)
        auroc,auprc,y_score = get_ml_auroc_auprc(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
        n_rank_score = get_auc_by_tpr(tpr_arr=get_tpr_by_n_expriments(predicted_vals=y_score,y_test=y_test,n=1000))
        tps_tns = get_tp_tn(y_test=y_test,y_train=y_train)
        print(f"Ith: {i+1}\{len(result_dataframes)} split is done")
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



def generate_features_labels_sql(splited_guide_data,target_column,off_target_column,y_column,manager):
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    for val in splited_guide_data.values():
        x_data = get_features(val,only_seq_info=ONLY_SEQ_INFO,target_column=target_column,off_target_column=off_target_column,manager=manager)
        x_data_all.append(x_data)
        y_labels_all.append(val[[y_column]].values)
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
    tp_ratio = tp_test / (tp_test + tn_test)
    return (tp_ratio,tp_test,tn_test,tp_train,tn_train)

'''funcion to run logsitic regression model and return roc,prc'''
def get_ml_auroc_auprc(X_train, y_train, X_test, y_test): # to run timetest unfold "#"
    # time_test(X_train,y_train)
    # exit(0)
    # train
    classifier = get_classifier()
    classifier.fit(X_train, y_train)
    # predict probs
    #y_pos_scores_probs = classifier.predict(X_test)
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
    

def get_classifier():
    if ML_TYPE == "LOGREG":
        return LogisticRegression(random_state=42,n_jobs=-1)
    elif ML_TYPE == "SVM":
        return SVC(kernel="linear",random_state=42)
    elif ML_TYPE == "RANDOMFOREST":
        return RandomForestClassifier(random_state=42,n_jobs=-1)
def enforce_seq_length(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long
'''get x_axis features for ml algo.
data - data frame for guiderna
only_seq_info - bolean for only seq or other features.'''
def get_features(data, only_seq_info,target_column,off_target_column,manager):
    bigwig_info = np.ones((data.shape[0],ENCODED_LENGTH))
    seq_info = np.ones((data.shape[0], ENCODED_LENGTH))
    # siteseq, negative target seq
    for index, (otseq, grnaseq, chrom, start, end) in enumerate(zip(data[off_target_column], data[target_column], data[CHROM], data[START], data[END])):
        otseq = enforce_seq_length(otseq, 23)
        grnaseq = enforce_seq_length(grnaseq, 23)
        otseq = otseq.upper()
        seq_info[index] = seq_to_one_hot(otseq, grnaseq)
        if IF_BP:
            bigwig_info[index] = bws_to_one_hot(file_manager=manager,chr=chrom,start=start,end=end)
            print (bigwig_info[index])
        seq_info[index] = seq_info[index] + bigwig_info[index]
        print(seq_info[index])
    # if bigwig"
    
    if only_seq_info:
        x = seq_info
    else:
        x = data[FEATURES_COLUMNS].values 
        x = np.append(x, seq_info, axis = 1)
    return x
def bws_to_one_hot(file_manager, chr, start, end):
    # back to original bp presantation
    indexing = BP_PRESENTATION - file_manager.get_number_of_bigiwig()
    epi_one_hot = np.zeros(ENCODED_LENGTH,dtype=float) # set epi feature with zeros
    for i_file,path in enumerate(file_manager.get_bigwig_paths()):
        bw_file = pyBigWig.open(path)
        values = bw_file.values(chr, start, end) # get values of base pairs in the coordinate
        for index, val in enumerate(values):
            # index * BP =  set index position 
            # indexing + i_file the gap between bp_presenation to each file slot.
            epi_one_hot[(index * BP_PRESENTATION) + (indexing + i_file)] = val 
    return epi_one_hot

def bw_to_one_hot(chr, start, end, bigwig_data, num_of_data):
    epi_one_hot = np.zeros(ENCODED_LENGTH,dtype=float) # set epi feature with zeros
    values = bigwig_data.values(chr, start, end) # get values of base pairs in the coordinate
    for index, val in enumerate(values):
        epi_one_hot[((index + 1) * BP_PRESENTATION) - num_of_data] = val # index + 1 - 1 based.
        # num of data - from j + 1 to amount of data + 1 
    return epi_one_hot
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
                onehot[BP_PRESENTATION * i + key] = 1
            if seq_guide[i] == base:
                onehot[BP_PRESENTATION * i + key] = 1
        if sequence[i] != seq_guide[i]:  # Mismatch
            try:
                if bases.index(sequence[i]) < bases.index(seq_guide[i]):
                    onehot[BP_PRESENTATION * i + 4] = 1
                else:
                    onehot[BP_PRESENTATION * i + 5] = 1
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
def set_bp_coding(management):
    global ENCODED_LENGTH,BP_PRESENTATION,IF_BP
    if_bp = input("press y/Y if bp coding is intended\n")
    if if_bp.lower() == "y":
        BP_PRESENTATION = 6 + management.get_number_of_bigiwig()
        ENCODED_LENGTH = 23 * BP_PRESENTATION
        IF_BP = True
def create_path_list(combined_folder):
    path_list = []
    for combined_file in os.listdir(combined_folder):
        combined_path = os.path.join(combined_folder,combined_file)
        path_list.append(combined_path)
    return path_list   
if __name__ == "__main__":
     ONLY_SEQ_INFO = set_if_seq()
     ML_TYPE = input("Please enter ML type: (LOGREG, SVM, XGBOOST, RANDOMFOREST):\n")
     management = File_management("pos","neg","bed","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig")
     set_bp_coding(management)
    # run_leave_one_out(sys.argv[1],sys.argv[2])
     crisprsql(data_table=sys.argv[1],target_colmun=TARGET,off_target_column=OFFTARGET,y_column="Label",file_name="Changeseq_csgs",file_manager= management)
    #get_tpr_by_n_expriments(predicted_vals=np.array([0.2,0.8,0.7,0.6,0.5]),y_test=np.array([0,1,0,1,0]),n=5)
    
