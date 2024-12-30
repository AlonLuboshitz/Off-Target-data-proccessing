'''
This is a module to create feature and labels from data.
'''

import pandas as pd
import numpy as np
import os
from pybedtools import BedTool
from sklearn.utils import shuffle
import itertools
from features_and_model_utilities import get_encoding_parameters
ALL_INDEXES = [] # Global variable to store indexes of data points when generating features

## FUNCTIONS:
# 1.
'''Args: 
1. whole data table - positives and negativs
Function: takes the data table, create a unique list of gRNAs, Split the data into seperate data frames
Based on gRNA
Outputs: 1. Dictionray - {gRNA : Data frame} 2. unique gRNA set
'''
def create_data_frames_for_features(data, if_data_reproducibility, target_column, exclude_guides = None, test_on_other_data = False):
    data_table = pd.read_csv(data) # open data
    if exclude_guides: # exlucde not empty
        if not test_on_other_data: # 
            data_table = return_df_without_guides(data_table, exclude_guides, target_column)
    # set unquie guide identifier, sorted if reproducibilty is need with data spliting
    if if_data_reproducibility:
        guides = sorted(set(data_table[target_column])) 
    else : 
        guides = list(set(data_table[target_column]))
        guides = shuffle(guides)
        # Create a dictionary of DataFrames, where keys are gRNA names and values are corresponding DataFrames
    df_dict = {grna: group for grna, group in data_table.groupby(target_column)}
    # Create separate DataFrames for each gRNA in the set
    result_dataframes = {grna: df_dict.get(grna, pd.DataFrame()) for grna in guides}
    return (result_dataframes, guides)

def return_df_without_guides(data_frame, guide_to_exlucde, data_frame_column):
    '''
    Return a dataframe without the guides in guides_to_exclude.
    Args:
        data_frame: A dataframe containing the data
        guide_to_exclude: (Tuple) (guides_description, path to guides to exclude from the data, target_columns)
    '''
    description, path, target_columns = guide_to_exlucde
    guides_to_exclude = set()
    guides_data = pd.read_csv(path)
    for column in target_columns:
        guides_to_exclude.update(guides_data[column].dropna().unique())  # Remove NaN values and add unique guides
    
    # Return the dataframe without the excluded guides
    return data_frame[~data_frame[data_frame_column].isin(guides_to_exclude)]
    
def generate_features_and_labels(data_path, manager, if_bp, if_only_seq , 
                                 if_seperate_epi, epigenetic_window_size, features_columns, if_data_reproducibility,
                                 columns_dict, transform_y_type = False, sequence_coding_type = 1, if_bulges = False,
                                 exclude_guides = None, test_on_other_data = False):
    '''
    This function generates x and y data for gRNAs and their corresponding off-targets.
    For each (gRNA, OTS) pair it one-hot encodes the sequences and adds epigenetic data if required.
    For each pair it will return their corresponding y values (1/0/read_count).
    It iterates on each gRNA : Data frame and extract the data.
    Uses internal functions for seq encoding, epigentic encoding, and bp intersection of epigenetics with seq.
    Args:
    1. Data path - path to the OTS data file.
    2. File manager instance to get epigentic files and their data. 
    3. Encoded length - the length of the one-hot encoded sequence.
    4. bp_presenation - the vector size for each base pair as.
    5. if_bp - boolean to add epigenetic data for each base pair.
    6. if_only_seq - boolean to use only sequence encoding.
    7. if_seperate_epi - boolean to use epigenetic data in seperate vector.
    8. epigenetic_window_size - the size of the window for epigenetic data.
    9. features_columns - list of columns to add as features.
    10. if_data_reproducibility - boolean to sort the data for reproducibility.
    11. columns_dict - dictionary of columns to use in the data frame with the columns:
    TARGET_COLUMN, OFFTARGET_COLUMN, CHROM_COLUMN, START_COLUMN, END_COLUMN,
    BINARY_LABEL_COLUMN, REGRESSION_LABEL_COLUMN, Y_LABEL_COLUMN
    12. transform_y_type - the type of transformation to apply to the y values.
    13. sequence_coding_type - the type of sequence encoding to use -  defualt is 1 - PiCRISPR style. 2 -  nuc*nuc per base pair.
    14. if_bulges - boolean to include bulges in the sequence encoding.
    15. exclude_guides - (tuple) (guides_description, path to guides to exclude from the data, target_column)
    16. test_on_other_data - boolean - if True dont exclude guides from that data
    Returns:
    1. x_data_all - list of x data for each gRNA.
    2. y_labels_all - list of y labels for each gRNA.
    3. guides - list of unique gRNAs.
    
'''
    splited_guide_data,guides = create_data_frames_for_features(data_path, if_data_reproducibility,
                                                                columns_dict["TARGET_COLUMN"],exclude_guides,test_on_other_data)
    x_data_all = []  # List to store all x_data
    y_labels_all = []  # List to store all y_labels
    ALL_INDEXES.clear() # clear indexes
    seq_len,nuc_num = get_encoding_parameters(sequence_coding_type,if_bulges) # get sequence encoding parameters
    encoded_length = seq_len * nuc_num # set encoded length
    for guide_data_frame in splited_guide_data.values(): # for every guide get x_features by booleans
        # get seq info - represented in all!
        if sequence_coding_type == 1: # PiCRISPR style
            seq_info = get_seq_one_hot(data=guide_data_frame, encoded_length = encoded_length, bp_presenation = nuc_num,
                                    off_target_column=columns_dict["OFFTARGET_COLUMN"],
                                    target_column=columns_dict["REALIGNED_COLUMN"]) #int8
        elif sequence_coding_type == 2: # Full encoding
            seq_info = full_one_hot_encoding(dataset_df=guide_data_frame, n_samples=len(guide_data_frame), seq_len=seq_len, nucleotide_num=nuc_num,
                                  off_target_column=columns_dict["OFFTARGET_COLUMN"], target_column=columns_dict["REALIGNED_COLUMN"])

        if if_bp: # epigentic value for every base pair in gRNA
            big_wig_data = get_bp_for_one_hot_enconded(data = guide_data_frame, encoded_length = encoded_length, manager = manager,
                                                        bp_presenation = nuc_num, chr_column = columns_dict["CHROM_COLUMN"],
                                                        start_column = columns_dict["START_COLUMN"], end_column = columns_dict["END_COLUMN"])
            seq_info = seq_info.astype(np.float32) 
            x_data = seq_info + big_wig_data
           
        elif if_seperate_epi: # seperate vector to epigenetic by window size
            epi_window_data = get_seperate_epi_by_window(data = guide_data_frame, epigenetic_window_size = epigenetic_window_size, 
                                                         manager = manager, chr_column = columns_dict["CHROM_COLUMN"],
                                                         start_column = columns_dict["START_COLUMN"])
            seq_info = seq_info.astype(np.float32)
            x_data = np.append(seq_info, epi_window_data ,axis=1)
        elif if_only_seq:
            x_data = seq_info
        else : # add features into 
            x_data = guide_data_frame[features_columns].values
            x_data = np.append(seq_info, x_data, axis = 1)
        if "Index" in guide_data_frame.columns:
            ALL_INDEXES.append(guide_data_frame["Index"])
        else:
            ALL_INDEXES.append(guide_data_frame.index)
        x_data_all.append(x_data)
        
        y_labels_all.append(guide_data_frame[[columns_dict["Y_LABEL_COLUMN"]]].values) # add label values by extracting from the df by series values.
    del splited_guide_data # free memory
    
    if transform_y_type:
        y_labels_all = transform_labels(y_labels_all, transform_y_type)
    
    return (x_data_all,y_labels_all,guides)

    

     
'''Args: 1. gRNA : data frame, 2. encoded length, 3. the vector size for each base pair as bp_repsentation
Function: init ones vector with amount of data points each encoded length size
Each data point sent to one hot encoding with (OTS seq, gRNA seq)'''
def get_seq_one_hot(data, encoded_length, bp_presenation, off_target_column, target_column):
    seq_info = np.ones((data.shape[0], encoded_length),dtype=np.int8)
    for index, (otseq, grnaseq) in enumerate(zip(data[off_target_column], data[target_column])):
        otseq = enforce_seq_length(otseq, 23)
        grnaseq = enforce_seq_length(grnaseq, 23)
        otseq = otseq.upper()
        seq_info[index] = partial_one_hot_enconding(otseq, grnaseq,encoded_length,bp_presenation)
    return seq_info
'''Args: 1. gRNA : data frame, 2. encoded length, 3. file manager with epigenetic files
 4. the vector size for each base pair as bp_repsentation
Function: init ones vector with amount of data points each encoded length size
Each data point sent to epigenetic encoding (add peak values for every base pair in grna location)'''
def get_bp_for_one_hot_enconded(data, encoded_length, manager, bp_presenation, chr_column, start_column, end_column):
    bigwig_info = np.ones((data.shape[0],encoded_length))
    for index, (chrom, start, end) in enumerate(zip(data[chr_column], data[start_column], data[end_column])):
        if not (end - start) == 23:
            end = start + 23
        bigwig_info[index] = bws_to_one_hot(file_manager=manager,chr=chrom,start=start,end=end,encoded_length=encoded_length,bp_presenation=bp_presenation)
    bigwig_info = bigwig_info.astype(np.float32)
    return bigwig_info
'''Args: 1. gRNA : data frame, 2. encoded length - as epi window size , 3. file manager with epigenetic files
Function: init ones vector with amount of data points each encoded length size times the amount of epigenetic files
i.e. (N_points, window_size + window_size) for 2 files.
Each data point sent to epigenetic encoding (add peak values for every base pair) where window size/2 added from ots location'''
def get_seperate_epi_by_window(data, epigenetic_window_size, manager, chr_column, start_column):
    epi_data = np.ones((data.shape[0],epigenetic_window_size * manager.get_number_of_bigiwig())) # set empty np array with data points and epigenetics window size
    for file_index, (bw_epi_name, bw_epi_file) in enumerate(manager.get_bigwig_files()): # get one or more files 
        #glb_max = manager.get_global_max_bw()[bw_epi_name] # get global max all over bigwig
        filler_start = file_index * epigenetic_window_size
        filler_end = (file_index + 1) * epigenetic_window_size
        for index, (chrom, start) in enumerate(zip(data[chr_column], data[start_column])):
        
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
def partial_one_hot_enconding(sequence, seq_guide,encoded_length,bp_presenation):
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

def full_one_hot_encoding(dataset_df, n_samples, seq_len, nucleotide_num, off_target_column, target_column):
    """
    Creates a one-hot encoding of sgRNA and off-target sequences.

    Args:
        dataset_df (pd.DataFrame): Dataset Dataframe. Must contain columns "SG_RNA_SEQ" and "OFF_TARGET".
        n_samples (int): Total number of samples in the dataset.
        seq_len (int): Length of the sequences.
        nucleotide_num (int): Number of distinct nucleotides (5 when including bulges).

    Returns:
        np.ndarray: One-hot encoded array, shape: (n_samples, seq_len, nucleotide_num ** 2)
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    one_hot_arr = np.zeros((n_samples, seq_len, nucleotide_num, nucleotide_num), dtype=np.int8)
    for i, (sg_rna_seq, off_seq) in enumerate(zip(dataset_df[target_column], dataset_df[off_target_column])):
        if len(off_seq) != len(sg_rna_seq):
            raise ValueError("len(off_seq) != len(sg_rna_seq)")
        actual_seq_size = len(off_seq)
        if actual_seq_size > seq_len:
            raise ValueError("actual_seq_size > seq_len")

        size_diff = seq_len - actual_seq_size
        for j in range(seq_len):
            if j >= size_diff:
                # note that it is important to take (sg_rna_seq_j, off_seq_j) as old models did the same.
                matrix_positions = nucleotides_to_position_mapping[(sg_rna_seq[j-size_diff], off_seq[j-size_diff])]
                one_hot_arr[i, j, matrix_positions[0], matrix_positions[1]] = 1
    # reshape to [n_samples, seq_len, nucleotide_num**2]
    one_hot_arr = one_hot_arr.reshape((n_samples, seq_len, nucleotide_num**2))
    one_hot_arr = one_hot_arr.reshape(n_samples, -1) # flatten the array
    return one_hot_arr

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


def create_nucleotides_to_position_mapping():
    """
    Creates a mapping of nucleotide pairs (sgRNA, off-target) to their numerical positions.
    This mapping includes positions for "N" nucleotides (representing any nucleotide).

    Returns:
        dict: A dictionary where keys are tuples of nucleotides ("A", "T"), ("G", "C"), etc.,
              and values are tuples representing their (row, column) positions in a matrix.
    """
    # matrix positions for ("A","A"), ("A","C"),...
    # tuples of ("A","A"), ("A","C"),...
    nucleotides_product = list(itertools.product(*(["ACGT-"] * 2)))
    # tuples of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in itertools.product(*(["01234"] * 2))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ("N","A"), ("N","C"),...
    n_mapping_nucleotides_list = [("N", char) for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]

    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    # tuples of ("A","N"), ("C","N"),...
    n_mapping_nucleotides_list = [(char, "N") for char in ["A", "C", "G", "T", "-"]]
    # list of tuples positions corresponding to ("A","A"), ("C","C"), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ["A", "C", "G", "T", "-"]]
    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    return nucleotides_to_position_mapping
## Data and features manipulation
'''Function to extract the guides indexes given which guides to keep.
The guides to keep are given as a list of indexes.
The function take the guides_indexes and return from ALL_INDEXES and spesific guide indexes'''
def get_guides_indexes(guide_idxs):
    choosen_indexes = [index for idx in guide_idxs for index in ALL_INDEXES[idx]]
    choosen_indexes = np.array(choosen_indexes)
    return choosen_indexes

### Transformations ###
def transform_labels(y_vals, transform_type):
    '''Transform the y values based on the given transformation type.
    The y_vals can be 1d np array or list of 1d np arrays.
    Returns the transformed y values.'''
    transform_type = transform_type.lower()
    if transform_type == "log":
        return log_transformation(y_vals)
    else:
        raise ValueError("Invalid transformation type.")
def log_transformation(y_vals):
    '''Conduct log transformation on the y values.
    The y_vals can be 1d np array or list of 1d np arrays.
    Returns the transformed y values.'''
    if isinstance(y_vals, list):
        transformed = [np.log(y_val + 1) for y_val in y_vals]
        for index,array in enumerate(transformed):
            if array.size != y_vals[index].size:
                raise ValueError("The size of the log transformed array does not match the original array.")
        return transformed
    else:
        transformed = np.log(y_vals + 1)
        if len(transformed) != len(y_vals):
            raise ValueError("The size of the log transformed array does not match the original array.")
        return transformed
    

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

def extract_features(X_train,encoded_length):
    '''
    This function splits one hot encoding into seq encoding and features encoding.
    '''
    seq_only = X_train[:, :encoded_length].astype(np.int8)
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

    
def keep_indexes_per_guide(data_frame = None, target_column = None):
    '''
    This function returns a list of indexes for each guide in the data frame.
    Args:
    1. data_frame: (str -path/ panda df) - The data frame containing the data.
    2. target_column: (str) - The name of the target column.
    Returns:
    A dictionary where keys are the guide names and values are the indexes of the data points.
    '''
    if isinstance(data_frame, str):
        data_frame = pd.read_csv(data_frame)
    elif not isinstance(data_frame, pd.DataFrame):
        raise ValueError("The data_frame must be a string or a pandas DataFrame.")
    guides = data_frame[target_column].unique() # create a unique list of guides
    if "Index" in data_frame.columns:
        guides_indexes = {guide: data_frame[data_frame[target_column] == guide]["Index"].values for guide in guides}
    else:
        guides_indexes = {guide: data_frame[data_frame[target_column] == guide].index for guide in guides}
    return guides_indexes
