import pandas as pd
import numpy as np
import os
import pybedtools
import re
'''for table from changeseq both same amount of columns merge togther.
set label column with active - 1 for guideseq expriments and 0 for change seq'''
def label_pos_neg(guideseq,negatives,output_name,target_column):
    
    guideseq = read_to_df_add_label(guideseq,1) # set 1 for guide seq
    negatives = read_to_df_add_label(negatives,0) # set 0 for changeseq
    negatives = remove_unmatching_guides(positive_data=guideseq,target_column=target_column,negative_data=negatives)
    before_gs = len(guideseq)
    before_cs = len(negatives)
    drop_on_columns = ["chrom","chromStart","chromEnd","offtarget_sequence","distance"] # columns to drop duplicates on
    guideseq, gs_duplicates = drop_by_colmuns(guideseq,drop_on_columns,"first")
    negatives,cs_duplicates = drop_by_colmuns(negatives,drop_on_columns,"first")
    neg_length=len(negatives)
    merged_data = pd.concat([guideseq,negatives])
    print(f"data points should be: {len(merged_data)}, {len(guideseq) + neg_length}")
    merged_data,md_duplicates = drop_by_colmuns(merged_data,drop_on_columns,"first")
    count_ones = sum(merged_data["Label"] == 1)
    count_zeros = sum(merged_data["Label"]==0)
    print(f"positives: {before_gs} - {gs_duplicates} = {before_gs-gs_duplicates}(gsb-gsd), label: {count_ones}")
    print(f"negatives: {neg_length} - {count_ones} = {neg_length - count_ones}(csb-csd), label: {count_zeros} ")
    print(merged_data.columns)
    # set index to first column
    merged_data.to_csv(output_name + ".csv",index=False)

def remove_unmatching_guides(positive_data,target_column,negative_data):
    # create a unuiqe set of guides from positive guides and negative guides
    positive_guides = set(positive_data[target_column])
    negative_guides = set(negative_data[target_column])
    # keep only the guides that presnted in the negative but not positive set
    diffrence_set = negative_guides - positive_guides
    intersect = negative_guides.intersection(positive_guides)
    print(f'intersect: {len(intersect)}, length pos: {len(positive_guides)}, length negative: {len(negative_guides)},\ndifrence: {len(diffrence_set)}')
    # remove the raws data from the nagative set matching the left guides
    before = len(negative_data)
    for guide in diffrence_set:
        negative_data = negative_data[negative_data[target_column]!=guide]
        after = len(negative_data)
        print(f'{before-after} rows were removed')
        before =  after
    return negative_data
'''drop duplicates by columns return data and amount of duplicates'''
def drop_by_colmuns(data,columns,keep):
    length_before = len(data)
    data = data.drop_duplicates(subset=columns,keep=keep)
    length_after = len(data)
    return data,length_before-length_after  
'''add label column to data frame'''
def read_to_df_add_label(path,label):
    table = pd.read_csv(path, sep=",",encoding='latin-1',on_bad_lines='skip')
    columns_before = table.columns
    table["Label"] = label
    columns_after = table.columns
    print(f"columns before: {columns_before}\nColumns after: {columns_after}")
    print(table.head(5))
    return table
def transofrm_casofiner_into_csv(path_to_txt):
    columns = ['target','Chrinfo','chromStart','offtarget_sequence','strand','distance']
    new_out = path_to_txt.replace(".txt",".csv")
    try:
        negative_file = pd.read_csv(path_to_txt,sep="\t",encoding='latin-1',on_bad_lines='skip')
    except pd.errors.EmptyDataError as e:
        print(f"{path_to_txt}, is empty")
        exit(0)
    negative_file.columns = columns
    potential_ots = add_info_to_negative(negative_file)
    potential_ots.to_csv(new_out,sep=',',index=False)
'''add info to casofinder output to get same columns ad guideseq'''
def add_info_to_negative(data):
    data['chrinfo_extracted'] = data['Chrinfo'].str.extract(r'(chr[^\s]+)') # extract chr
    data = data.rename(columns={ 'chrinfo_extracted':'chrom'}) 
    data = data.drop('Chrinfo',axis=1) # drop unwanted chrinfo
    data['chromEnd'] = data['chromStart'] + 23 # assuming all guide are 23 bp
    data['casofinder_readc'] = 0 # set readcount to zero
    data = upper_case_series(data,colum_name="offtarget_sequence") # upcase all oftarget in data
    ordered_columns = ['chrom','chromStart','chromEnd','casofinder_readc','','strand','offtarget_sequence','','distance','target',''] # order of data
    new_data = pd.DataFrame(columns=ordered_columns) # create new data with spesific columns
    for column in data.columns:
        new_data[column] = data[column] # set data in the new df from the old one
    return new_data
def upper_case_series(data,colum_name):
    print(data.head(5))
    values_list = data[colum_name].values
    print(values_list[:5])
    upper_values = [val.upper() for val in values_list]
    print(upper_values[:5])
    data[colum_name] = upper_values
    print(data.head(5))
    return data
'''function to return a list of columns for the bed epigenetic file and put column 5 with score'''
def get_bed_columns(bedtool):
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
    # get data
    whole_data_bed = pybedtools.BedTool.from_dataframe(whole_data)
    epigentic_data = pybedtools.BedTool(epigentic_data)
    # set columns to data columns + bed columns (5th of bed is score)
    columns = whole_data.columns.tolist() 
    columns = columns +  get_bed_columns(epigentic_data)
    intersection_wa = whole_data_bed.intersect(epigentic_data,wb=True,s=if_strand) # wb keep both information
    intersection_df_wa = intersection_wa.to_dataframe(header=None,names = columns)
    return whole_data,intersection_df_wa
    
def assign_epigenetics(data,intersection,file_ending,chrom_type):
    chrom_column = f'{chrom_type}_{file_ending}' # set chrom type and mark column
    # set binary and score column with zeros
    binary_column = f'{chrom_column}_binary'
    score_column = f'{chrom_column}_score'
    fold_column = f'{chrom_column}_fold_enrichemnt'
    log_fold_column = f'{chrom_column}_log_fold_enrichemnt'
    logp_column = f'{chrom_column}_logp'
    logq_column = f'{chrom_column}_logq'
    data[binary_column] = 0 
    # data[score_column] = 0
    # data[fold_column] = 0 
    # data[logp_column] = 0
    # data[logq_column] = 0
    data[log_fold_column] = 0
    # convert score data from intersection info to list
    # score_vals = intersection["score"].tolist()
    fold_vals= intersection["fold_enrichemnt"].tolist()
    # logp_vals = intersection["logp"].tolist()
    # logq_vals = intersection["logq"].tolist()
    log_fold_vals = np.array(fold_vals)
    log_fold_vals = np.log(log_fold_vals)
    if not intersection.empty:
        try:
            print(data.head(5))
            # assign intersection indexes with 1
            data.loc[intersection["Index"], binary_column] = 1
            # # assign intersection indexes with score values
            # data.loc[intersection["Index"], score_column] = score_vals
            # data.loc[intersection["Index"], fold_column] = fold_vals
            # data.loc[intersection["Index"], logp_column] = logp_vals
            # data.loc[intersection["Index"], logq_column] = logq_vals
            data.loc[intersection["Index"], log_fold_column] = log_fold_vals
            print(data.head(5))
            
           
        except KeyError as e:
              print(data,': file has no intersections output will be with 0')
        #     # # : input is dismissed via running this as subprocess
        #     #input("press anything to continue: ")
    labeled_epig_1 = sum(data[binary_column]==1)
    labeled_epig_0 =  sum(data[binary_column]==0)
    print(f"length of intersect: {len(intersection)}, amount of labled epigenetics: {labeled_epig_1}")
    print(f'length of data: {len(data)}, 0: {labeled_epig_0}, 1+0: {labeled_epig_1 + labeled_epig_0}')
    return data
def get_ending(txt):
    ending = txt.split("/")[-1].split(".")[0]
    return ending
def run_intersection(merged_data_path,bed_folder,if_update):
    data = pd.read_csv(merged_data_path)
    data["Index"] = data.index # set index column
    bed_types_nd_paths = get_bed_folder(bed_folder)
    new_data_name = merged_data_path.replace(".csv","")
    new_data_name = f'{new_data_name}_withEpigenetic_log.csv'
    if if_update:
        bed_types_nd_paths = remove_exsiting_epigenetics(data,bed_types_nd_paths) # remove exsiting epigenetics
        new_data_name = merged_data_path # update exsiting data
    for chrom_type,bed_paths in bed_types_nd_paths:
        for bed_path in bed_paths:
            data,intersect = intersect_with_epigentics(data,epigentic_data=bed_path,if_strand=False)
            data = assign_epigenetics(data=data,intersection=intersect,file_ending=get_ending(bed_path),chrom_type=chrom_type)
    data = data.drop("Index", axis=1) # remove "Index" column
    data.to_csv(new_data_name,index=False)

def remove_exsiting_epigenetics(data,bed_type_nd_paths):
    new_chrom_information = [] # assign new list to keep only new data
    for chrom_type,bed_paths in bed_type_nd_paths:
        paths_list = [] 
        for bed_path in bed_paths:
            file_ending = get_ending(bed_path) 
            column_to_check = f'{chrom_type}_{file_ending}'
            if not column_to_check in data.columns: # if the name is in the columns data already been made.
                paths_list.append(bed_path)
        new_chrom_information.append((chrom_type,paths_list))
    return new_chrom_information


''' function iterate on bed folder and returns a list of tuples:
each tuple: [0] - folder name [1] - list of paths for the bed files in that folder.'''
def get_bed_folder(bed_parent_folder):
    # create a list of tuples - each tuple contain - folder name, folder path inside the parent bed file folder.
    subfolders_info = [(entry.name, entry.path) for entry in os.scandir(bed_parent_folder) if entry.is_dir()]
    # Create a new list of tuples with folder names and the information retrieved from the get bed files
    result_list = [(folder_name, get_bed_files(folder_path)) for folder_name, folder_path in subfolders_info]
    return result_list

'''function retrives bed files
args- bed foler
return list paths.'''
def get_bed_files(bed_files_folder):
    bed_files = []
    for foldername, subfolders, filenames in os.walk(bed_files_folder):
        for name in filenames:
            # check file type the narrow,broad, bed type. $ for ending
            if re.match(r'.*(\.bed|\.narrowPeak|\.broadPeak)$', name):
                bed_path = os.path.join(foldername, name)
                bed_files.append(bed_path)
    return bed_files
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

           




if __name__ == "__main__":
    #transofrm_casofiner_into_csv("/home/alon/masterfiles/pythonscripts/Changeseq/one_output.txt")
    #label_pos_neg("/home/alon/masterfiles/pythonscripts/Changeseq/GUIDE-seq.csv","/home/alon/masterfiles/pythonscripts/Changeseq/CHANGE-seq.csv",output_name="merged_csgs",target_column="target")
    #run_intersection(merged_data_path="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs.csv",bed_folder="/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics",if_update=False)
   
    change = pd.read_csv("/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_withEpigenetic.csv")
    change =  remove_buldges(change,"offtarget_sequence")
    change.to_csv("/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_withEpigenetic.csv",index=False)