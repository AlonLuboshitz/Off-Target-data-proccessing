import pandas as pd
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

def intersect_with_epigentics(whole_data,epigentic_data,if_strand):
    
    whole_data_bed = pybedtools.BedTool.from_dataframe(whole_data)
    epigentic_data = pybedtools.BedTool(epigentic_data)
    intersection_wa = whole_data_bed.intersect(epigentic_data,wa=True,s=if_strand)
    intersection_df_wa = intersection_wa.to_dataframe(header=None,names = whole_data.columns.tolist())
    return whole_data,intersection_df_wa
    
def assign_epigenetics(data,intersection,file_ending,chrom_type):
    chrom_column = f'{chrom_type}_{file_ending}'
    data[chrom_column] = 0
    if not intersection.empty:
        try:
            # index keeps original index of data frame of combined file
            # assign intersection indexes with 1
            data.loc[intersection["Index"], chrom_column] = 1
            # set the corresponding index of the bed file in new column for correlation analysis.
           
        except KeyError as e:
              print(data,': file has no intersections output will be with 0')
        #     # # : input is dismissed via running this as subprocess
        #     #input("press anything to continue: ")
    labeled_epig_1 = sum(data[chrom_column]==1)
    labeled_epig_0 =  sum(data[chrom_column]==0)
    print(f"length of intersect: {len(intersection)}, amount of labled epigenetics: {sum(data[chrom_column]==1)}")
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
    new_data_name = f'{new_data_name}_withEpigenetic.csv'
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

           




if __name__ == "__main__":
    #transofrm_casofiner_into_csv("/home/alon/masterfiles/pythonscripts/Changeseq/one_output.txt")
    label_pos_neg("/home/alon/masterfiles/pythonscripts/Changeseq/GUIDE-seq.csv","/home/alon/masterfiles/pythonscripts/Changeseq/one_output.csv",output_name="merged_csgs_casofinder",target_column="target")
    run_intersection(merged_data_path="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_casofinder.csv",bed_folder="/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics",if_update=False)
