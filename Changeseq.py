import pandas as pd
import numpy as np
import os
import pybedtools
import re
import time


def get_bed_columns(bedtool):
    '''This function accepts a bedtool and returns the columns of the bedtool as a list.
    The function add the score, fold_enrichement, logp and logq columns to the list.
    Args:
    1. bedtool - a bedtool object
    ------------
    Returns: a list of columns'''
    # Get the first interval
    first_interval = next(iter(bedtool))
    # Get the number of fields (columns) for the first interval
    num_fields = len(first_interval.fields)
    columns = []
    for i in range(num_fields):
            columns.append(i+1)
    columns[4] = "score"
    columns[6] = "fold_enrichemnt"
    columns[7] = "logp"
    columns[8] = "logq"
    return columns
def intersect_with_epigentics(whole_data,epigentic_data,if_strand):
    '''This function intersects off-target data with epigenetic data/data from bed file.
    The functions accepts two data frames - whole_data and epigentic_data.
    It intersects the point by -wb param for bed intersection function.
    Args:
    1. whole_data - data frame with off-target data
    2. epigentic_data - bed file with epigenetic data
    3. if_strand - boolean, if True, the function will intersect by strand
    ------------
    Returns: whole_data, intersection_df_wa - data frame with off-target data and data frame with intersection data.'''
    # get data
    whole_data_bed = pybedtools.BedTool.from_dataframe(whole_data)
    epigentic_data = pybedtools.BedTool(epigentic_data)
    # set columns to data columns + bed columns (5th of bed is score)
    columns = whole_data.columns.tolist() 
    columns = columns +  get_bed_columns(epigentic_data)
    intersection_wa = whole_data_bed.intersect(epigentic_data,wb=True,s=if_strand) # wb keep both information
    intersection_df_wa = intersection_wa.to_dataframe(header=None,names = columns)
    return whole_data,intersection_df_wa
    
def assign_epigenetics(off_target_data,intersection,file_ending,chrom_type,score_type_dict={"binary":True}):
    '''This function assign epigenetic data to the off-target data frame.
    It extracts the epigenetic type - i.e. chromatin accesesibility, histone modification, etc.
    To that it adds the epigenetic mark itself - i.e. H3K4me3, H3K27ac, etc.
    To each combination of type and mark it assigns a binary column by defualt.
    If other score types are given it will assign them as well.
    An example to set binary value and the fold enrichement value:
    score_type_dict = {"binary":True,"score":False,"fold_enrichemnt":True,"log_fold_enrichemnt":False,"logp":False,"logq":False}
    Args:
    1. off_target_data - data frame with off-target data
    2. intersection - data frame with intersection data
    3. file_ending - the ending of the bed file - i.e. H3K4me3, H3K27ac, etc.
    4. chrom_type - the type of the epigenetic data - i.e. chromatin accesesibility, histone modification, etc. 
    5. score_type_dict - a dictionary with the score types and the values to assign to the columns
    ------------
    Returns: off_target_data - data frame with the epigenetic data assigned.
    '''
    chrom_column = f'{chrom_type}_{file_ending}' # set chrom type and mark column
    columns_dict = {key: f'{chrom_column}_{key}' for key in score_type_dict.keys()} # set columns names
    # add columns to the off-target data
    for column_name in columns_dict.values():
        off_target_data[column_name] = 0
    # Set a dictionary with the columns names and the intersect values
    values_dict = {key: None for key in score_type_dict.keys()} # Intersect columns
    log_gold_flag = False
    for key in values_dict.keys():
        if key == "log_fold_enrichemnt": # if log fold enrichemnt need to be set set the flag.
            log_gold_flag = True
            continue
        values_dict[key] = intersection[key].tolist()
    # set log fold enrichemnt
    if log_gold_flag and "fold_enrichemnt" in values_dict.keys():
        log_fold_vals = np.array(values_dict["fold_enrichemnt"])
        log_fold_vals = np.log(log_fold_vals)
    
    if not intersection.empty:
        try:
            print(f"Assigning the next epigenetic values: {columns_dict.keys()}")
            print("OT data before assignment:\n",off_target_data.head(5))
            time.sleep(1)
            # Assign intersection indexes in the off-target data with 1 for binary column and values to other columns
            for key in score_type_dict.keys():
                if key == "binary":
                    off_target_data.loc[intersection["Index"], columns_dict["binary"]] = 1
                else :
                    off_target_data.loc[intersection["Index"], columns_dict[key]] = values_dict[key]
            print("OT data after assignment:\n",off_target_data.head(5))
        except KeyError as e:
              print(off_target_data,': file has no intersections output will be with 0')
    ## Print statistics   
    labeled_epig_1 = sum(off_target_data[columns_dict["binary"]]==1)
    labeled_epig_0 =  sum(off_target_data[columns_dict["binary"]]==0)
    if (labeled_epig_1 + labeled_epig_0) != len(off_target_data):
        raise RuntimeError("The amount of labeled epigenetics is not equal to the amount of data")
    print(f"length of intersect: {len(intersection)}, amount of labled epigenetics: {labeled_epig_1}")
    print(f'length of data: {len(off_target_data)}, 0: {labeled_epig_0}, 1+0: {labeled_epig_1 + labeled_epig_0}')
    return off_target_data

def get_ending(txt):
    ending = txt.split("/")[-1].split(".")[0]
    return ending
def run_intersection(merged_data_path,bed_folder,if_update):
    '''This function intersect off-target data with given folder of epigenetic data given in bed files.
    It will intersect the data with each bed file in the folder and assign the epigenetic data to the off-target data.
    If if_update is True, the function will update the existing data with the new epigenetic data.
    Args:
    1. merged_data_path - path to the merged off-target data
    2. bed_folder - path to the folder with the epigenetic data
    3. if_update - boolean, if True the function will update the existing data with the new epigenetic data.
    ----------
    Returns: None
    Saves the new data frame with the epigenetic data in the same path as the merged data with the ending _withEpigenetic.csv'''
    data = pd.read_csv(merged_data_path)
    data["Index"] = data.index # set index column
    bed_types_nd_paths = get_bed_folder(bed_folder)
    new_data_name = merged_data_path.replace(".csv","")
    new_data_name = f'{new_data_name}_withEpigenetic.csv'
    if if_update:
        bed_types_nd_paths = remove_exsiting_epigenetics(data,bed_types_nd_paths,True) # remove exsiting epigenetics
        new_data_name = merged_data_path # update exsiting data
    for chrom_type,bed_paths in bed_types_nd_paths:
        for bed_path in bed_paths:
            data,intersect = intersect_with_epigentics(data,epigentic_data=bed_path,if_strand=False)
            data = assign_epigenetics(off_target_data=data,intersection=intersect,file_ending=get_ending(bed_path),chrom_type=chrom_type)
    data = data.drop("Index", axis=1) # remove "Index" column
    data.to_csv(new_data_name,index=False)

def remove_exsiting_epigenetics(data,bed_type_nd_paths,full_match=False):
    '''This function accpets data frame and list of tuples where
    First element is epigenetic type - Chrom, methylation, etc
    Second element is the epigeneitc mark itself - h3k4me3...
    Removes from the tuple list any epigenetic mark that already exists in the data frame.
    if full_match is True, the function will remove only the epigenetic marks that are fully matched in the data frame.
    if full_match is False, the function will remove any epigenetic mark that is partially matched in the data frame.
    '''
    new_chrom_information = [] # assign new list to keep only new data
    for chrom_type,bed_paths in bed_type_nd_paths:
        paths_list = [] 
        for bed_path in bed_paths:
            file_ending = get_ending(bed_path) 
            if full_match:
                column_to_check = f'^{chrom_type}_{file_ending}$'
            else : column_to_check = f'^{chrom_type}_{file_ending}'
            if any(re.match(column_to_check, column) for column in data.columns):
                continue
            else : paths_list.append(bed_path)
            
        new_chrom_information.append((chrom_type,paths_list))
    return new_chrom_information




'''Keep just chroms info from the type chrN where N is 1-23,X,Y,M'''
def remove_random_alt_chroms(data,chrom_column):
    before = len(data)
    print(data[chrom_column].head(5))
    # match(r'^chr([1-9]|1[0-9]|2[0-3]|[XYM])_.*$'
    filtered_df = data[data[chrom_column].str.match(r'^chr([1-9]|1[0-9]|2[0-3]|[XYM])$')]
    print(filtered_df[chrom_column].head(5))
    print(f"before fil: {before}, after: {len(filtered_df)}")
    
    return filtered_df
def remove_buldges(data,off_target_column):
    '''This function remove bulges from the data given the off target column
    removes by length and by "-" in the off target column.
    ------
    returns a data frame without bulges'''
    before = len(data)
    print(data[off_target_column].head(5))
    # keep only ots with 23 length
    data = data[data[off_target_column].str.len() == 23]
    print(data[off_target_column].head(5))
    after= len(data)
    print(f"before fil: {before}, after: {(after)}, removed : {before-after}")
    before = after
    # remove ots with "-"
    data = data[data[off_target_column].str.find("-") == -1]
    after = len(data)
    print(f"before fil: {before}, after: {(after)}, removed : {before-after}")
    return data

def count_feature(data_frame,feature_column,label_column):
    '''Given a data frame, feature column and label column
    Return the amount of label > 0 and feature column > 0 - i.e. positive and feature is present
    Return the amount of label == 0 and feature column > 0 - i.e. negative and feature is present'''
    pos_feature = len(data_frame[(data_frame[label_column] > 0) & (data_frame[feature_column] > 0)])
    neg_feature = len(data_frame[(data_frame[label_column] == 0) & (data_frame[feature_column] > 0)])
    return (pos_feature,neg_feature)

def count_features(data_path, feature_columns, label_column):
    '''Given a data path, feature columns and label column:
    For each feature column count the amount of positive and negative labels where the 
    feature is present using count_feature function
    '''
    data = pd.read_csv(data_path) # open data
    feature_amount_dict = {key : count_feature(data,key,label_column) for key in feature_columns}
    return feature_amount_dict    


def code_for_turing_read_into_labels():
    
    print(new_gs.head(5))
    print(new_gs.info())
    # print amount of label =1 or 0
    print(sum(new_gs["Align.#Bulges"] > 0))
    print(sum(new_gs["Label"] > 0))
    new_gs["Read_count"] = new_gs["Label"]
    # Turn the label column into binary
    new_gs['Label'] = new_gs['Label'].apply(lambda x: 1 if x > 0 else 0)
    print(sum(new_gs["Label"] > 0))
    print(new_gs.head(5))
    print(new_gs["Label"].nunique())
    print(new_gs.info())
    print(new_gs["Label"].value_counts())
    print(sum(new_gs["Read_count"] > 0))
    new_gs = new_gs[new_gs["Align.#Bulges"] == 0]
    print(sum(new_gs["Align.#Bulges"] > 0))
    new_gs=remove_buldges(new_gs,"offtarget_sequence")
    print(sum(new_gs["Align.#Bulges"] > 0))
    print(new_gs["Label"].nunique())
    print(new_gs["Label"].value_counts())


if __name__ == "__main__":
    #transofrm_casofiner_into_csv("/home/alon/masterfiles/pythonscripts/Changeseq/one_output.txt")
    #label_pos_neg("/home/alon/masterfiles/pythonscripts/Changeseq/GUIDE-seq.csv","/home/alon/masterfiles/pythonscripts/Changeseq/CHANGE-seq.csv",output_name="merged_csgs",target_column="target")
    #run_intersection(merged_data_path="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs.csv",bed_folder="/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics",if_update=False)
    # 1.transform casofinder into csv 
    # 2.   label data
    #3. add epigenetics
    
    #label_pos_neg("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/only_pos_new_guideseq.csv",
     #             "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/only_pos_changeseq.csv",output_name="merged_new_csgs",output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/",target_column="target")
   
    pd1 = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/40_exp/40.csv")
    pd2 = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/50_exp/50.csv")
    sg1 = set(pd1['target'])
    sg2 = set(pd2['target'])
    sgRNA = sg1.union(sg2)
    print(len(sgRNA))
    print(sgRNA)
    sgRNA_list = list(sgRNA)
    with open("/home/dsi/lubosha/Off-Target-data-proccessing/Data/sgRNAcaso.txt", "w") as file:
        for sgRNA in sgRNA_list:
            file.write(sgRNA + "6 \n")