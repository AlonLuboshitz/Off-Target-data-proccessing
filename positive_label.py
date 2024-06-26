# code for labeling positive (off target chromosomal positions by guideseq).
import pandas as pd
import os
import re
import numpy as np
import pybedtools
import time
from utilities import validate_path, create_folder, remove_dir_recursivly, get_bed_folder

ORDERED_COLUMNS = ['chrom','chromStart','chromEnd','Position','Filename','strand','offtarget_sequence','target','realigned_target','Read_count','missmatches','insertion','deletion','bulges','Label']

#### Identified guideseq preprocessing functions ####

def process_folder(input_folder):
    '''
Function gets a folder with guide-seq identified txt files.
It creates a new output folder (if not exists) with csv files filtered by label identified function
folder name created: _labeled
Args:
1. input_folder - folder with identified txt files
------------
Returns: None
Runs: identified_to_csv function on each txt file in the folder'''
    label_output_folder = input_folder + '_labeled'
    create_folder(label_output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(input_folder, filename)
            identified_to_csv(txt_file_path, label_output_folder)

def identified_to_csv(input_path,output_path):
    
    '''given an path for an indentified.txt, filter by bool the BED_site_name culom.
    BED_site_name colom if not null represents a recognisble off-target site.
    1. filter by bed site column and mismatches up to 6.
    2. Get Off target sites with missmatches only/ bulges and missmatches.
    3. Extract file name - expriment name.
    4. Create csv file in the output path named with expirement name + "_label"
    Columns kept are: 
    chrom, chromStart, chromEnd, Position, Filename, strand, offtarget_sequence, target,
    realigned_target, Read_count, missmatches, insertions, deletions, bulges, Label
    Args:
    1. input_path - path to the identified.txt file
    2. output_path - path to the output folder
    ------------
    Returns: None
    Saves: csv file in the output folder with the columns mentioned above.
   '''

    identified_file = pd.read_csv(input_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
    # 1. filter by bed_site - accautal off target site
    valid_otss = identified_file.dropna(subset=['BED_Site_Name'])
    # 2. filter by missmatch count <=6
    valid_otss = valid_otss[
        (valid_otss['Site_SubstitutionsOnly.NumSubstitutions']  <= 6) |
        (valid_otss['Site_GapsAllowed.Substitutions'] <= 6)  ]
    mismatch_df = get_ots_missmatch_only(valid_otss) # Get mismatch only
    bulge_df = get_ots_bulges_nd_mismatches(valid_otss) # Get bulges
    merged_df = pd.concat([mismatch_df,bulge_df],axis=0,ignore_index=True) # Merged both pds
    # 3. get expirement name
    exp_name = merged_df.iloc[0]['Filename']
    exp_name = exp_name.replace('.sam','')
    output_filename = exp_name + "_labeled.csv"
    output_path = os.path.join(output_path,output_filename)
    # 5. create csv file
    merged_df.to_csv(output_path, index=False)
    print("created {} file in folder: {}".format(output_filename,output_path))

def get_ots_missmatch_only(guideseq_identified_data_frame):
    '''This function accepets guide seq idenetified data frame
    It extract Off target sites with missmatches only
    Setting insertions, deletions, bulges to 0.
    Args:
    1. guideseq_identified_data_frame - data frame with guide-seq identified data
    ------------
    Returns: mismatch_df - data frame with off-target sites with missmatches only
    '''
    # Drop rows without ots seq in mismatch
    guideseq_identified_data_frame = guideseq_identified_data_frame.dropna(subset=['Site_SubstitutionsOnly.Sequence'])
    columns = {'WindowChromosome':'chrom' ,'Site_SubstitutionsOnly.Start':'chromStart','Site_SubstitutionsOnly.End' : 'chromEnd',
               'Position':'Position','Filename':'Filename','Site_SubstitutionsOnly.Strand':'strand',
               'Site_SubstitutionsOnly.Sequence':'offtarget_sequence','TargetSequence':'target',
               'RealignedTargetSequence':'realigned_target','bi.sum.mi':'Read_count',
               'Site_SubstitutionsOnly.NumSubstitutions':'missmatches'}
    mismatch_df = guideseq_identified_data_frame[columns.keys()]
    mismatch_df.rename(columns=columns,inplace=True)
    mismatch_df[["insertion","deletion","bulges"]] = 0
    mismatch_df['Label'] = 1
    return mismatch_df

def get_ots_bulges_nd_mismatches(guideseq_identified_data_frame):
    '''This function accepets guide seq idenetified data frame
    It extract Off target sites with bulges and mismatches
    Args:
    1. guideseq_identified_data_frame - data frame with guide-seq identified data
    ------------
    Returns: bulge_df - data frame with off-target sites with bulges and mismatches
     '''
    columns = {'WindowChromosome':'chrom' ,'Site_GapsAllowed.Start':'chromStart','Site_GapsAllowed.End':'chromEnd',
               'Position':'Position','Filename':'Filename','Site_GapsAllowed.Strand':'strand',
               'Site_GapsAllowed.Sequence':'offtarget_sequence','TargetSequence':'target',
               'RealignedTargetSequence':'realigned_target','bi.sum.mi':'Read_count',
               'Site_GapsAllowed.Substitutions':'missmatches',
               'Site_GapsAllowed.Insertions':'insertion','Site_GapsAllowed.Deletions': 'deletion'}
    # Drop rows without bulges
    guideseq_identified_data_frame = guideseq_identified_data_frame.dropna(subset=['Site_GapsAllowed.Sequence'])
    bulge_df = guideseq_identified_data_frame[columns.keys()]
    bulge_df.rename(columns=columns,inplace=True)
    bulge_df["bulges"] = bulge_df["insertion"] + bulge_df["deletion"]
    bulge_df['Label'] = 1
    return bulge_df



def merge_positives(folder_path, n_duplicates, file_ending, output_folder_name):
        
    '''
    Function looks for multiple duplicates from the same exprimenet and merge their data.
    Mergning will be the summation of the read count for the same sites!
    NOTE: all files should have the same ending, for example: -D(n)_labeled
    Function gets each file by iterating on the number of duplicates and changing the file ending.
    For each 2 or more duplicates concate the data.
    Args:
     1. folder_path - path to the folder with the labeled files
     2. n_duplicates - number of duplicates for each expriment
     3. file_ending - ending of the file name
     4. output_folder_name - name of the output folder
     ------------
     Returns: None
     Saves: csv files in the output folder with the merged data'''
    assert n_duplicates > 1, f"duplicates should be more then 1 per expriment, got: {n_duplicates}"
    # more then 1 duplicate
    file_names = os.listdir(folder_path)
    # create pattern of xxx-D(n) + suffix
    pattern = r"(.+?-D)\d+" + re.escape(file_ending)
    # create one string from all the file names and find the matching pattern.
    file_names = ''.join(file_names)
    mathces = re.findall(pattern,file_names)
    # use a set to create a unique value for n duplicates
    unique = list(set(mathces))
    print('before mergning: ',len(mathces))
    print('after mergning: ',len(unique))
    # get tuple list - df, name
    final_file_list = mergning(unique,n_duplicates,file_ending,folder_path)   
    # create folder for output combined expriment:
    # remove .csv\.txt from ending.
    file_ending = file_ending.rsplit('.', 1)[0]
    # create folder
    output_path = os.path.join(os.path.dirname(folder_path), output_folder_name)
    create_folder(output_path)
    # create csv file from eached grouped df.
    for tuple in final_file_list:
        name = tuple[1] + '.csv'
        temp_output_path = os.path.join(output_path,name)
        tuple[0].to_csv(temp_output_path,index=False)
    
        


def mergning(files, n_duplicates, file_ending, folder_path):
    '''
    This function gets a list of files and merge them togther summing the read count.
    It merge n duplicates for each file.
    Args:
    1. files - list of files to merge
    2. n - number of duplicates for each file
    3. file_ending - ending of the file name
    4. folder_path - path to the folder with the files
    ------------
    Returns: final_file_list - list of tuples with the merged data frames and the file name''' 
    assert n_duplicates > 1, f"duplicates should be more then 1 per expriment, got: {n_duplicates}"  
    # more then 1 duplicate per file
    final_file_list = []
    grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target', 'realigned_target','missmatches','insertion', 'deletion','bulges' ]
    for file_name in files:
        # create n duplicates file list
        n_file_list =[]
        for count in range(int(n_duplicates)):
            # create string for matching duplicate file
            temp_file_name = file_name + str(count+1) + file_ending 
            input_path = os.path.join(folder_path,f'{temp_file_name}')
            # append df into the list
            n_file_list.append(pd.read_csv(input_path,sep=",",encoding='latin-1',on_bad_lines='skip'))
        # n_file_list have all duplicates in it, merge them:
        merged_df = pd.concat(n_file_list, axis=0, ignore_index=True)
        print ('before grouping: ',len(merged_df))
        # group by position, number of missmatches, and siteseq. sum the bi - reads
        grouped_df = merged_df.groupby(grouping_columns).agg({
    'Position': 'first', 'Filename': 'first', 'Read_count': 'sum', 
    'Label': 'first'  
}).reset_index()
        
        print ('after grouping: ',len(grouped_df))
        grouped_df = grouped_df[ORDERED_COLUMNS]
        # append df to final list
        final_file_list.append((grouped_df,file_name)) 
    return final_file_list

def concat_data_frames(folder_path = None, first_df = None, second_df = None):
    '''Function concat data frames vertically.
    If folder path is given it will concat all the csv files in the folder.
    Else it will concat two data frames given in the first and second paths.
    Args:
    1. folder_path - path to the folder with the csv files
    2. first_df - first data frame to concat
    3. second_df - second data frame to concat
    ------------
    Returns: data frame with the concatenated data frames'''
    if folder_path:
        files = os.listdir(folder_path)
        data_frames = []
        for file in files:
            file_path = os.path.join(folder_path,file)
            data_frames.append(pd.read_csv(file_path))
        return pd.concat(data_frames, axis = 0, ignore_index = True)
    elif (not first_df.empty) and (not second_df.empty):
        return pd.concat([first_df,second_df], axis = 0, ignore_index = True)
    else: 
        raise ValueError("No data frames given to concat.")

def preprocess_identified_files(folder_path, idenetified_folder_name, n_duplicates, output_data_name, erase_sub_dir = False):
    '''Function preprocess identified files in a given folder:
    1. Turn idenetified single files into csv files using identified_to_csv function.
    2. Merge the same experiments with multiple duplicates using merge_positives function.
    3. Concat all the data frames into one data frame.
    4. Save the data frame in a csv file.
    5. If erase_sub_dir is True, the function will remove the sub directories.
    Args:
    1. folder_path - path to the folder with the identified files
    2. idenetified_folder_name - name of the folder with the identified files
    3. n_duplicates - number of duplicates for each experiment
    4. output_data_name - name of the output data file
    5. erase_sub_dir - boolean, if True the function will remove the sub directories
    ------------
    Returns: None'''
    if validate_path(folder_path):
        identified_folder_path = os.path.join(folder_path, idenetified_folder_name)
        if validate_path(identified_folder_path):
            process_folder(identified_folder_path)
            labled_folder_path  = identified_folder_path + "_labeled"
            if not validate_path(labled_folder_path): # validate label folder was created
                raise RuntimeError(f"No labeled folder created need to process: {idenetified_folder_name} agian!")
            if n_duplicates > 1: # no duplicates need to be merged
                out_put_merged_folder_name = "merged_experiments"
                merge_positives(labled_folder_path, n_duplicates, '_labeled.csv', out_put_merged_folder_name)
                merged_folder_path = os.path.join(folder_path, out_put_merged_folder_name)
                if not validate_path(merged_folder_path): # validate merged folder was created
                    raise RuntimeError(f"No merged folder created need to process: {idenetified_folder_name} agian!")
            else : 
                merged_folder_path = labled_folder_path
            
            if not output_data_name.endswith('.csv'): # add .csv to output file if needed.
                output_data_name = output_data_name + '.csv'
            merged_df = concat_data_frames(folder_path = merged_folder_path) # concat all the data frames
            merged_df.to_csv(os.path.join(folder_path, output_data_name), index = False) # save merged data frame
            if erase_sub_dir:
                remove_dir_recursivly(labled_folder_path) # remove labeled folder
                if n_duplicates > 1:
                    remove_dir_recursivly(merged_folder_path)
    print(f"Preprocessing of {idenetified_folder_name} is done. Merged data saved in {output_data_name} file.")
          

### Negative Labeling functions ###
'''Example input file (DNA bulge size 2, RNA bulge size 1):
/var/chromosomes/human_hg38
NNNNNNNNNNNNNNNNNNNNNRG 2 1
GGCCGACCTGTCGCTGACGCNNN 5'''
# inputpath for folder containing indentified files.
# extract guide sequence and create intput file out of it.
def create_csofinder_input_by_identified(input_path,output_path):
    # get targetseq = the guide rna used
    identified_file = pd.read_csv(input_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
    seq = identified_file.iloc[0]['TargetSequence']
    # guideseq size 'N' string
    n_string = 'N' * len(seq)
    exp_name = seq + "_" + identified_file.iloc[0]['Filename']
    output_filename = f"{seq}_input.txt"
    output_path = os.path.join(output_path,output_filename)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as txt_file:
            txt_file.write("/home/labhendel/Documents/cas-offinder_linux_x86-64/hg38noalt\n")
            txt_file.write(n_string + "\n")
            txt_file.write(seq + ' 6')

'''function to read a table and get the unique target grna to create an input text file
for casofinder
table - all data
target_colmun - column to get the data
outputname - name of folder to create in the scripts folder'''
def create_cas_ofinder_inputs(table , target_column, output_name, path_for_casofiner):
    table = pd.read_excel(table)
    output_path = os.getcwd() # get working dir path
    output_path = os.path.join(output_path,output_name) # add folder name to it
    create_folder(output_path)
    try:
        guides = set(table[target_column]) # create a set (unquie) guides
    except KeyError as e:
        print(f"no column: {target_column} in data set, make sure you entered the right one.")
        exit(0)
    casofinder_path = get_casofinder_path(path_for_casofiner)
    casofinder_path = casofinder_path + "\n"
    one_file_path = os.path.join(output_path,f"one_input.txt")
    n_string = 'N' * 23
    with open(one_file_path,'w') as file:
        file.write(casofinder_path)
        file.write(n_string + "\n")
    for guide in guides:
        n_string = 'N' * len(guide)
        output_filename = f"{guide}_input.txt"
        temp_path = os.path.join(output_path,output_filename)
        with open(temp_path, 'w') as txt_file:
            txt_file.write(casofinder_path)
            txt_file.write(n_string + "\n")
            txt_file.write(guide + ' 6')
        with open(one_file_path,'a') as txt:
            txt.write(guide + ' 6\n')
    
'''if genome == hg19,hg38 set own path else keep others.'''
def get_casofinder_path(genome):
    path = genome
    if genome == "hg19":   
        path = "/home/labhendel/Documents/cas-offinder_linux_x86-64/hg19"
    elif genome == "hg38":
        path = "/home/labhendel/Documents/cas-offinder_linux_x86-64/hg38noalt"
    else : print(f"no genome file exists for: {genome}")
    return path
''' function to return a dict with the lengths of the guides'''
def guides_langth(guide_set):
    lengths = {}
    for guide in guide_set:
        length_g = len(guide) # length of guide
        if length_g in lengths.keys():
            lengths[length_g] += 1
        else:   lengths[length_g] = 1
    return lengths
     



### Cas-offinder creation and negative labeling functions ###

def transform_casofiner_into_csv(path_to_txt):
    columns = ['target','Chrinfo','chromStart','offtarget_sequence','strand','missmatches']
    output_path = path_to_txt.replace(".txt",".csv")
    try:
        negative_file = pd.read_csv(path_to_txt,sep="\t",encoding='latin-1',on_bad_lines='skip')
    except pd.errors.EmptyDataError as e:
        print(f"{path_to_txt}, is empty")
        exit(0)
    negative_file.columns = columns
    print(negative_file.head(5))
    print(negative_file.info())
    potential_ots = add_info_to_casofinder_file(negative_file)
    potential_ots.to_csv(output_path,sep=',',index=False)

def add_info_to_casofinder_file(data):

    '''add info to casofinder output file:
    1. Extract chrinfo from Chrinfo column.
    2. Set chromend to be chrom start + len of the ots
    3. Set Read_count, insertions, deletions, bulges, Position to 0
    4. Set realigned target to be the same as target
    5. Set Filename to empty string
    6. Upcase offtarget_sequence'''

    data['chrinfo_extracted'] = data['Chrinfo'].str.extract(r'(chr[^\s]+)') # extract chr
    data = data.rename(columns={ 'chrinfo_extracted':'chrom'}) 
    data = data.drop('Chrinfo',axis=1) # drop unwanted chrinfo
    print(data.head(5))
    print(data.info())
    data['chromEnd'] = data['chromStart'] + data['offtarget_sequence'].str.len() 
    print(data.head(5))
    print(data.info())
    #data['chromEnd'] = data['chromStart'] + 23 # assuming all guide are 23 bp
    data['Read_count'] = data['insertion'] = data['deletion'] = data['bulges'] = data['Position'] = data["Label"] = 0 
    data['realigned_target'] = data['target'] # set realigned target to target
    data['Filename'] = '' # set filename to empty
   
    data = upper_case_series(data,colum_name="offtarget_sequence") # upcase all oftarget in data
    print(data.head(5))
    print(data.info())
    new_data = pd.DataFrame(columns=ORDERED_COLUMNS) # create new data with spesific columns
    for column in data.columns:
        new_data[column] = data[column] # set data in the new df from the old one
    print(new_data.head(5))
    print(new_data.info())
    return new_data



def upper_case_series(data, colum_name):
    '''function to upcase a series in a data frame'''
    values_list = data[colum_name].values
    upper_values = [val.upper() for val in values_list]
    data[colum_name] = upper_values
    return data


### Merge positive data set with negative data set ###

def merge_positive_negative(positives, negatives, output_name, output_path, target_column, remove_bulges):
    '''Function to merge positive data set and negative data set:
    1. Add label column to both data sets with 1 for positive and 0 for negative.
    2. Remove from the negative data all the guides that presnted in the positive data.
    3. Group by columns and aggregate Read_count
    4. Print the amount of data points in the merged data set.
    5. Save the data set'''    
    positives = read_to_df_add_label(positives,1) # set 1 for guide seq
    negatives = read_to_df_add_label(negatives,0,True) # set 0 for changeseq
    negatives = remove_unmatching_guides(positive_data=positives,target_column=target_column,negative_data=negatives)
    grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target','missmatches','insertion', 'deletion','bulges' ]

    if remove_bulges:
        negatives = negatives[negatives['bulges'] == 0]
        positives = positives[positives['bulges'] == 0]
        grouping_columns = ['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target','missmatches' ]
    
    positives, gs_duplicates = drop_by_colmuns(positives,grouping_columns,"first")
    negatives,cs_duplicates = drop_by_colmuns(negatives,grouping_columns,"first")
    neg_length = len(negatives)
    pos_length = len(positives)
    merged_data = pd.concat([positives,negatives])
    print(f"data points should be: Merged: {len(merged_data)},Pos + Neg: {pos_length+ neg_length}")
    merged_data,md_duplicates = drop_by_colmuns(merged_data,['chrom', 'chromStart', 'chromEnd','strand', 'offtarget_sequence', 'target','missmatches'],"first")
     
    
    
    
    count_ones = sum(merged_data["Label"] > 0)
    count_zeros = sum(merged_data["Label"]==0)
    print(f"Positives: {pos_length}, By label: {count_ones}")
    print(f"Negatives: {neg_length} - {count_ones} = {neg_length - count_ones + (count_ones-md_duplicates)}, label: {count_zeros} ")
    print(merged_data.columns)
    output_path = f"{os.path.join(output_path,output_name)}.csv"
    merged_data.to_csv(output_path,index=False)

def remove_unmatching_guides(positive_data, target_column, negative_data):
    '''Function to remove from the negative data all the guides that presnted in the positive data.
    1. Create a unuiqe set of guides from positive guides and negative guides.
    2. Keep only the guides that presnted in the negative but not positive set.
    3. Remove the guides from the negative set'''
    # create a unuiqe set of guides from positive guides and negative guides
    positive_guides = set(positive_data[target_column])
    negative_guides = set(negative_data[target_column])
    # keep only the guides that presnted in the negative but not positive set
    diffrence_set = negative_guides - positive_guides
    intersect = negative_guides.intersection(positive_guides)
    print(f'intersect: {len(intersect)}, length pos: {len(positive_guides)}, length negative: {len(negative_guides)},\ndifrence: {len(diffrence_set)}')
   
    before = len(negative_data)
    for guide in diffrence_set:
        negative_data = negative_data[negative_data[target_column]!=guide]
        after = len(negative_data)
        guide_seq_amount = len(positive_data[positive_data[target_column]==guide])
        removed_amount = before-after
        print(f'{removed_amount} rows were removed')
        before =  after
    return negative_data

def drop_by_colmuns(data,columns,keep):
    '''drop duplicates by columns return data and amount of duplicates'''
    length_before = len(data)
    data = data.drop_duplicates(subset=columns,keep=keep)
    length_after = len(data)
    return data,length_before-length_after  

def read_to_df_add_label(path , label, if_negative=False):
    '''Add label column to data frame
    Column will be named Label and will be set to the label value
    If negative is True, the function will read the data as negative data and set the label to 0'''
    table = pd.read_csv(path, sep=",",encoding='latin-1',on_bad_lines='skip')
    columns_before = table.columns
    if (not "Label" in table.columns):
        table["Label"] = label
    columns_after = table.columns
    print(f"columns before: {columns_before}\nColumns after: {columns_after}")
    print(table.head(5))
    return table

### Epigenetic assignment ###
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
    values_dict.pop("binary") # remove binary from the dict
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
        values_dict["log_fold_enrichemnt"] = log_fold_vals.tolist()
    
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


'''
function gets path for identified (guideseq output data) folder and calls:
process_folder function, which creates csv folder named: identified_labeled_sub_only
this folder is then used in merge_positive function to merge the duplicate guide-seq expriments
argv 1 - path to identified.
argv 2 -  number of duplicated expriments
argv 3 - keep the identified label folder or erase it
'''
if __name__ == '__main__':
    #preprocess_identified_files("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/40_exp","identified",2,"40",True)
    #transform_casofiner_into_csv("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Negatives/sgRNA_caso_output.txt")
    #merge_positive_negative(positives="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_guideseq.csv",negatives="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/Negatives/sgRNA_caso_output.csv",output_name="merged_gs_caso",output_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab",target_column="target",remove_bulges=True)
   
    
    
    
    






  

