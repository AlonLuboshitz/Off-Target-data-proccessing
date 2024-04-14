'''
This is a class to handle processed data and transform it into features
For the machine learning algorithms
This includes libaries such: pandas, numpy, common_variables class'''

import pandas as pd
import numpy as np
import os
from pybedtools import BedTool
#from common_variables import 
from sklearn.utils import shuffle
from file_management import File_management
from constants import TARGET_COLUMN , OFFTARGET_COLUMN, CHROM_COLUMN, START_COLUMN, END_COLUMN, BINARY_LABEL_COLUMN
ALL_INDEXES = [] # Global variable to store indexes of data points when generating features
## in common variables class there are columns for: 
'''1. TARGET - gRNA seq column
   2. OFFTARGET- off-target seq column
   3. CHROM - chrom column
   4. START,END - start and end coordinates'''
## in common variables class there are booleans for:
'''1. Seq_only, 2. bigwig info with seq 3. seperate bigwig info channel 4. seq and 1 epi value'''

## FUNCTIONS:
# 1.
'''Args: 
1. whole data table - positives and negativs
Function: takes the data table, create a unique list of gRNAs, Split the data into seperate data frames
Based on gRNA
Outputs: 1. Dictionray - {gRNA : Data frame} 2. unique gRNA set
'''
def create_data_frames_for_features(data, if_data_reproducibility):
    data_table = pd.read_csv(data) # open data
    # set unquie guide identifier, sorted if reproducibilty is need with data spliting
    if if_data_reproducibility:
        guides = sorted(set(data_table[TARGET_COLUMN])) 
    else : 
        guides = list(set(data_table[TARGET_COLUMN]))
        guides = shuffle(guides)
        # Create a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames
    df_dict = {grna: group for grna, group in data_table.groupby(TARGET_COLUMN)}
    # Create separate DataFrames for each gRNA in the set
    result_dataframes = {grna: df_dict.get(grna, pd.DataFrame()) for grna in guides}
    return (result_dataframes, guides)

'''Args:
1. Dict - {gRNA : Data frame},  2. File manager instaces to get files and thier data 3. booleans for type of feature enconding
Function: Store x and y lists with x features after encoding, and coresponding y values (1/0/log(1+ read count))
Iterate on each gRNA : Data frame and extract the Data
Outputs --> x & y lists with N nd.arrays each of them is for gRNA (N - total number of gRNAs)
Uses - internal functions for seq encoding, epigentic encoding, and bp intersection of epigenetics with seq'''
def generate_features_and_labels(data_table, manager, encoded_length, bp_presenation, if_bp, if_only_seq , if_seperate_epi, epigenetic_window_size, features_columns, if_data_reproducibility):
    splited_guide_data,guides = create_data_frames_for_features(data_table, if_data_reproducibility)
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    ALL_INDEXES.clear() # clear indexes
    for guide_data_frame in splited_guide_data.values(): # for every guide get x_features by booleans
        # get seq info - represented in all!
        seq_info = get_seq_one_hot(data=guide_data_frame, encoded_length = encoded_length, bp_presenation = bp_presenation) #int8
        
        if if_bp: # epigentic value for every base pair in gRNA
            big_wig_data = get_bp_for_one_hot_enconded(data = guide_data_frame, encoded_length = encoded_length, manager = manager, bp_presenation = bp_presenation)
            seq_info = seq_info.astype(np.float32) 
            x_data = seq_info + big_wig_data
           
        elif if_seperate_epi: # seperate vector to epigenetic by window size
            epi_window_data = get_seperate_epi_by_window(data = guide_data_frame, epigenetic_window_size = epigenetic_window_size, manager = manager)
            seq_info = seq_info.astype(np.float32)
            x_data = np.append(seq_info, epi_window_data ,axis=1)
        elif if_only_seq:
            x_data = seq_info
        else : # add features into 
            x_data = guide_data_frame[features_columns].values
            x_data = np.append(seq_info, x_data, axis = 1)
        ALL_INDEXES.append(guide_data_frame.index)
        x_data_all.append(x_data)
        
        y_labels_all.append(guide_data_frame[[BINARY_LABEL_COLUMN]].values) # add label values by extracting from the df by series values.
    del splited_guide_data # free memory
    return (x_data_all,y_labels_all,guides)


'''Args: 1. gRNA : data frame, 2. encoded length, 3. the vector size for each base pair as bp_repsentation
Function: init ones vector with amount of data points each encoded length size
Each data point sent to one hot encoding with (OTS seq, gRNA seq)'''
def get_seq_one_hot(data, encoded_length, bp_presenation):
    seq_info = np.ones((data.shape[0], encoded_length),dtype=np.int8)
    for index, (otseq, grnaseq) in enumerate(zip(data[OFFTARGET_COLUMN], data[TARGET_COLUMN])):
        otseq = enforce_seq_length(otseq, 23)
        grnaseq = enforce_seq_length(grnaseq, 23)
        otseq = otseq.upper()
        seq_info[index] = seq_to_one_hot(otseq, grnaseq,encoded_length,bp_presenation)
    return seq_info
'''Args: 1. gRNA : data frame, 2. encoded length, 3. file manager with epigenetic files
 4. the vector size for each base pair as bp_repsentation
Function: init ones vector with amount of data points each encoded length size
Each data point sent to epigenetic encoding (add peak values for every base pair in grna location)'''
def get_bp_for_one_hot_enconded(data, encoded_length, manager, bp_presenation):
    bigwig_info = np.ones((data.shape[0],encoded_length))
    for index, (chrom, start, end) in enumerate(zip(data[CHROM_COLUMN], data[START_COLUMN], data[END_COLUMN])):
        if not (end - start) == 23:
            end = start + 23
        bigwig_info[index] = bws_to_one_hot(file_manager=manager,chr=chrom,start=start,end=end,encoded_length=encoded_length,bp_presenation=bp_presenation)
    bigwig_info = bigwig_info.astype(np.float32)
    return bigwig_info
'''Args: 1. gRNA : data frame, 2. encoded length - as epi window size , 3. file manager with epigenetic files
Function: init ones vector with amount of data points each encoded length size times the amount of epigenetic files
i.e. (N_points, window_size + window_size) for 2 files.
Each data point sent to epigenetic encoding (add peak values for every base pair) where window size/2 added from ots location'''
def get_seperate_epi_by_window(data, epigenetic_window_size, manager):
    epi_data = np.ones((data.shape[0],epigenetic_window_size * manager.get_number_of_bigiwig())) # set empty np array with data points and epigenetics window size
    for file_index, (bw_epi_name, bw_epi_file) in enumerate(manager.get_bigwig_files()): # get one or more files 
        #glb_max = manager.get_global_max_bw()[bw_epi_name] # get global max all over bigwig
        filler_start = file_index * epigenetic_window_size
        filler_end = (file_index + 1) * epigenetic_window_size
        for index, (chrom, start) in enumerate(zip(data[CHROM_COLUMN], data[START_COLUMN])):
        
            epi_data[index,filler_start:filler_end] = get_epi_data_bw(epigenetic_bw_file=bw_epi_file,chrom=chrom,center_loc=start,window_size=epigenetic_window_size,max_type = 1)
        print(epi_data[0])
    epi_data = epi_data.astype(np.float32)
    return epi_data
## ONE HOT ENCONDINGS:
'''1. (gRNA, OTS)
creating |encoded_length| vector 
first 4 values are for base paris: A,T,C,G
next 2 values are indicating which letter belongs to which sequence.
returned flatten vector'''
def seq_to_one_hot(sequence, seq_guide,encoded_length,bp_presenation):
    bases = ['A', 'T', 'C', 'G']
    onehot = np.zeros(encoded_length, dtype=np.int8) # init encoded length zeros vector (biary vec, int8)
    sequence_length = len(sequence)
    for i in range(sequence_length): # for each base pair 
        for key, base in enumerate(bases): # set by key of [A-0,T-1,C-2,G-3]
            if sequence[i] == base: # OTS
                onehot[bp_presenation * i + key] = 1 
            if seq_guide[i] == base: # gRNA
                onehot[bp_presenation * i + key] = 1
        if sequence[i] != seq_guide[i]:  # Mismatch
            try: # Set direction of mismatch
                if bases.index(sequence[i]) < bases.index(seq_guide[i]):
                    onehot[bp_presenation * i + 4] = 1
                else:
                    onehot[bp_presenation * i + 5] = 1
            except ValueError:  # Non-ATCG base found
                pass
    return onehot
'''1.2 Reverse one hot encoding to sequence'''
def reversed_ont_hot_to_seq(one_hot, bp_presenation):
    bases = ['A', 'T', 'C', 'G']
    sequence = ''
    guide_seq = ''
    for i in range(int(len(one_hot) / bp_presenation)):
        base_indices = np.nonzero(one_hot[i * bp_presenation:i * bp_presenation + 4])[0] # get indices of 1's
        # Check mismatchess
        if one_hot[i*bp_presenation + 4] == 1: # mismatch
            # First base is ots second is gRNA
            sequence += bases[base_indices[0]]
            guide_seq += bases[base_indices[1]]
        elif one_hot[i*bp_presenation + 5] == 1: # mismatch
             # First base is gRNA second is ots
            sequence += bases[base_indices[1]]
            guide_seq += bases[base_indices[0]]
        else : # no mismatch add to both sequences the same value
            sequence += bases[base_indices[0]]
            guide_seq += bases[base_indices[0]]
    return sequence, guide_seq

'''2. bigwig (base pair epigentics) to one hot
Fill vector sized |encoded length| with values from bigwig file. 

'''
def bws_to_one_hot(file_manager, chr, start, end,encoded_length,bp_presenation):
    # back to original bp presantation
    indexing = bp_presenation - file_manager.get_number_of_bigiwig()
    epi_one_hot = np.zeros(encoded_length,dtype=np.float32) # set epi feature with zeros
    try:
        for i_file,file in enumerate(file_manager.get_bigwig_files()):
            values = file[1].values(chr, start, end) # get values of base pairs in the coordinate
            for index,val in enumerate(values):
                # index * BP =  set index position 
                # indexing + i_file the gap between bp_presenation to each file slot.
                epi_one_hot[(index * bp_presenation) + (indexing + i_file)] = val
    except ValueError as e:
        return None
    return epi_one_hot

## Functions to obtain epienetic data for each base pair
'''
1. get epigenetic values from bigwig file and fill window size vector with those values'''

def get_epi_data_bw(epigenetic_bw_file, chrom, center_loc, window_size,max_type):
    positive_step = negative_step = int(window_size / 2) # set steps to window/2
    if (window_size % 2): # not even
        positive_step += 1 # set pos step +1 (being rounded down before)

        
    chrom_lim =  epigenetic_bw_file.chroms(chrom)
    indices = np.arange(center_loc - negative_step, center_loc + positive_step)
    # Clip the indices to ensure they are within the valid range
    indices = np.clip(indices, 0, chrom_lim - 1)
    # Retrieve the values directly using array slicing
    y_values = epigenetic_bw_file.values(chrom, indices[0], indices[-1] + 1)
    # Get local min and local max
    min_val = epigenetic_bw_file.stats(chrom,indices[0],indices[-1] + 1,type="min")[0] 
    if max_type: # None for local max, other 1 or global
        max_val = max_type
    else :
        max_val = epigenetic_bw_file.stats(chrom,indices[0],indices[-1] + 1,type="max")[0] 
        if max_val == 0.0: # max val is 0 then all values are zero
            return np.zeros(window_size,dtype=np.float32) 
    # Create pad_values using array slicing
    pad_values_beginning = np.full(max(0, positive_step - center_loc), min_val)
    pad_values_end = np.full(max(0, center_loc + negative_step - chrom_lim), min_val)

    # Combine pad_values with y_values directly using array concatenation
    y_values = np.concatenate([pad_values_beginning, y_values, pad_values_end])
    y_values = y_values.astype(np.float32)
    y_values[np.isnan(y_values)] = min_val # replace nan with min val
    y_values /= max_val # devide by max [local/global/1].
    return y_values

'''2. CREATE epigenetic data with 1/0 values via bed file (interval) information.
uses help function - update_y_values_by_intersect()'''

def get_epi_data_bed(epigenetic_bed_file, chrom, center_loc,window_size):
    positive_step = negative_step = int(window_size / 2) # set steps to window/2
    if (window_size % 2): # not even
        positive_step += 1 # set pos step +1 (being rounded down before)
    start = center_loc - negative_step # set start point
    end = center_loc + positive_step # set end point
    string_info = f'{chrom} {start} {end}' # create string for chrom,start,end
    ots_bed = BedTool(string_info,from_string=True) # create bed temp for OTS
    intersection = ots_bed.intersect(epigenetic_bed_file) 
    if not len(intersection) == 0: # not empty
        y = update_y_values_by_intersect(intersection, start, window_size)
    else : 
        y = np.zeros(window_size,dtype=np.int8)
    os.remove(intersection.fn)
    os.remove(ots_bed.fn)
    return y
    
## Assistant functions:
def enforce_seq_length(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long
'''help function for get_epi_by_bed'''
def update_y_values_by_intersect(intersect_tmp, start, window_size):
    print(intersect_tmp.head())
    y_values = np.zeros(window_size,dtype=np.int8) # set array with zeros as window size i.e 2000
    for entry in intersect_tmp:
        intersect_start = entry.start # second field (0-based) is start
        intersect_end = entry.end # 3rd fiels is end
    # get indexs for array values allways between 0 and window size
        if intersect_start == start:
            start_index = 0
        else : start_index = intersect_start - start - 1
        end_index = intersect_end - start - 1
    
        y_values[start_index:end_index] = 1 # set one for intersection range
    y_values[0] = 0 # for better respresnation - setting 0,0
    return y_values

## Data and features manipulation
'''Function to extract the guides indexes given which guides to keep.
The guides to keep are given as a list of indexes.
The function take the guides_indexes and return from ALL_INDEXES and spesific guide indexes'''
def get_guides_indexes(guide_idxs):
    choosen_indexes = [index for idx in guide_idxs for index in ALL_INDEXES[idx]]
    choosen_indexes = np.array(choosen_indexes)
    return choosen_indexes

'''given feature list, label list split them into
test and train data.
transform into ndarray and shuffle the train data'''
def order_data(X_feature,Y_labels,i,if_shuffle,if_print,sampler,if_over_sample):
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
    if if_over_sample:
        x_train,y_train = over_sample(x_train,y_train,sampler)
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


def over_sample(X_train, y_train, over_sampler):
    num_minority_samples_before = sum(y_train == 1)
    num_majority = sum(y_train==0)
    
    X_train, y_train = over_sampler.fit_resample(X_train,y_train)
    num_minority_samples_after = sum(y_train == 1)
    # Calculate the number of samples that have been duplicated
    num_samples_duplicated = num_minority_samples_after - num_minority_samples_before
    print(f"Number of samples duplicated: {num_samples_duplicated}\nclass ratio: {num_minority_samples_after / num_majority}")
    return (X_train, y_train)

'''function splits one hot encoding into seq encoding and features encoding.'''
def extract_features(X_train,encoded_length):
    seq_only = X_train[:, :encoded_length]
    features = X_train[:, encoded_length:]
    return [seq_only,features]




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
def get_duplicates(file_manager):
    x_features, y_labels, guides = generate_features_and_labels(file_manager.get_merged_data_path(), file_manager,
                                                                 23*6, 6, False, True, False, 2000, [], False)
    # Check if there are duplicates in x_features
    x_features = np.concatenate(x_features, axis=0)
    unique_ele,counts = np.unique(x_features, axis=0, return_counts=True)
    duplicates = unique_ele[counts > 1]
    unuiq = len(unique_ele)
    not_un = len(x_features)
    print("len(duplicates):", len(duplicates))
    print("not_un - unuiq:", not_un - unuiq)
    print("Unique elements:", unique_ele)
    print("Counts:", counts)
    otss = []
    if unuiq < not_un:
        print("There are duplicates in x_features")
        for ele in duplicates:
            otss.append(reversed_ont_hot_to_seq(ele, 6))
    # save otss to txt file
    with open("otss.txt", "w") as file:
        for ots in otss:
            file.write(f"OTS: {ots[0]}, gRNA: {ots[1]}\n")
if __name__ == "__main__":
    from Server_constants import EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_GS_EPI , DATA_PATH
    file_manager = File_management("", "", EPIGENETIC_FOLDER, BIG_WIG_FOLDER,CHANGESEQ_GS_EPI , DATA_PATH)
    x_features, y_labels, guides = generate_features_and_labels(file_manager.get_merged_data_path(), file_manager,
                                                                 23*6, 6, False, True, False, 2000, [], False)
    
    
    
