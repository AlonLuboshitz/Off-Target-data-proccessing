# script for getting a bed file with chromatin info and cas-offinder output.csv
# creating a label for the bed file expriment with 1 if intersects with the cas-offinder or 0 if not.
import os
import pandas as pd
import pybedtools
import sys
import subprocess
import re
from correlation_analysis import merge_files
'''function args: 1 - bed folder path
2 - path for guideseq folder, inside combined files needs chrom labeling

'''
def run_chrom_labeling(bed_parent_folder_path,guideseq_folder):
    # get list of tuples from get_bed_fodler function contains:
    # type of chrom info and list of paths to the correspoding bed files.
    chrom_type_path_list = get_bed_folder(bed_parent_folder_path)
    # get list of combined_output paths:
    combined_output_path_list = get_combined_paths(guideseq_folder)
    for combined_path in combined_output_path_list:
        # for each combined path run the script on:
        # retrive chrom type and list of bed files
        for chrom_type,path_list in chrom_type_path_list:
            # for each path in the bed files list run the script
            for bed_path in path_list:
                apply_for_folder(combined_path,bed_path,chrom_type)

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

           
''' function args:
1 guideseq folder
returns list of paths for combined_output folder.'''
def get_combined_paths(guideseq_folder):
    combined_out_put_path_list = []
    for folder,subfolders,filenames in os.walk(guideseq_folder):
        if 'combined_output' in subfolders:
            combined_output_path = os.path.join(folder,f'combined_output')
            combined_out_put_path_list.append(combined_output_path)
    return combined_out_put_path_list   

'''args: 
1. folder of combined_files - e.a labeled.
iterated on files and pass each one to opencrhom function
2. bed path - from run chrom label function.
3. chrom_info - name of folder with the bed files. e.a folder name openchrom consists bed files of
runs chrom labeling '''
def apply_for_folder(folder_combined_path,bed_path,chrom_info):
    # create a new folder named chrom info taged.
    output_path = os.path.join(os.path.dirname(folder_combined_path),f'chrom_info_tag')
    if not os.path.exists(output_path):
        os.mkdir(output_path) 
    # iterate combined files
    print ("labeling combined_output folder {}, with bed file: {}, from type: {}".format(folder_combined_path,bed_path,chrom_info))
    for combined_file in os.listdir(folder_combined_path):
        combined_path = os.path.join(folder_combined_path,combined_file)
        # label the combined file vie bed fils
        add_chrom_label(combined_path,bed_path,output_path,chrom_info) 

'''function args:
1. gets path for combined file - with active\inactive labeling
2. bed_path - bedfile for chromatin info e.g - accesability, methylation, phopholyration etc..
3. output_path - path to place new file with intersections with the bed file information.
4. chrom_info - folder name of type of chrom info - accsesbility, mythlation, phsos...
'''
def add_chrom_label(combined_path,bed_path,output_path,chrom_info):
    # true if manualy updating exsiting file - the combine path is already labled and equals the output path
    if not combined_path == output_path:
        # check if already the combined file has been labeled with difrrenet bed file:
        # retrive ending of file name 
        file_path_ending =  combined_path.split('/')[-1].split('.')[0] + "_chrom_labeled.csv"
        # merge the ending with the output folder
        output_path = os.path.join(output_path,file_path_ending)
        # if the file exists -> use this file instead of the original combined.
        if os.path.exists(output_path):
            combined_path = output_path
    # transfrom files into data frames with chr,strat,end,index 
    # get combined file to df.
    combined_data = pd.read_csv(combined_path)   
    combined_bed_data = create_from_data_bed_data_frame(combined_data,False)
    bed_data = create_from_bed_to_data_frame(bed_path,False)
    # Perform the intersection using pybedtools, -wo keeps information from both - e.a both columns and base pair amount for each intersection 
    combined_tool = pybedtools.BedTool.from_dataframe(combined_bed_data)
    bed_tool_data = pybedtools.BedTool.from_dataframe(bed_data)
    intersections = combined_tool.intersect(bed_tool_data, wo=True, s=False) 
    # concate btoh columns and pair amount
    intersect_colmuns = combined_bed_data.columns.tolist() + bed_data.columns.tolist() + ["pair_amount"]
    # convert intersection output into df.
    intersection_df = intersections.to_dataframe(header=None, names=intersect_colmuns)
    # Add a new column to combined_data based on the intersection result - 1 for intersect
    column_bed_filename = get_exp_name_to_column(bed_path,chrom_info)
    combined_data[column_bed_filename] = 0  
    # Add a new column to combined_data for bed file indexing for later corelation analysis
    bedindex_column = column_bed_filename + "_index"
    combined_data[bedindex_column] = 0
    combined_data[bedindex_column] = combined_data[bedindex_column].astype(float)
    # intersection is not empty
    if not intersection_df.empty:
        # transform bed indexes into a list for further assignment
        bed_index_list = intersection_df["IndexBed"].tolist()

        try:
            # index keeps original index of data frame of combined file
            # assign intersection indexes with 1
            combined_data.loc[intersection_df["Index"], column_bed_filename] = 1
            # set the corresponding index of the bed file in new column for correlation analysis.
            combined_data.loc[intersection_df["Index"], bedindex_column] = bed_index_list
        except KeyError as e:
              print(combined_path,': file has no intersections output will be with 0')
        #     # # : input is dismissed via running this as subprocess
        #     #input("press anything to continue: ")
 
    combined_data.to_csv(output_path, index=False)
''' create data frame from bed file
args is path to bed file
if strand - add strand information'''
def create_from_bed_to_data_frame(bed_path,if_strand):
    # Read BED file into a DataFrame bed file has no headers
    bed_data = pd.read_csv(bed_path, sep='\t', header=None)
    # Get number of columns
    num_columns = bed_data.shape[1]
    # Create column names based on the number of columns
    if if_strand:
        column_names = ['Chrbed', 'Startbed', 'Endbed','Strand'] + [str(i) for i in range(5, num_columns + 1)]
    else:
        column_names = ['Chrbed', 'Startbed', 'Endbed'] + [str(i) for i in range(4, num_columns + 1)]
    # Assign the new column names to the DataFrame
    bed_data.columns = column_names 
    # set index
    bed_data['IndexBed'] = bed_data.index
    return bed_data
''' create bed type data frame from combined file
arg is data frame
if_starnd = keep strand information'''   
def create_from_data_bed_data_frame(combined_data,if_strand):
    # convert combined df into bedfile with : Chr,Start,End,Index (strand, if strand)
    columns = ['chrinfo_extracted', 'Position']
    
    if if_strand:
        ordered_columns = ['Chr', 'Start', 'End', 'Index', 'Score', 'Strand']
        columns.append('Strand')
    csv_bed_data = combined_data[columns].copy()
    csv_bed_data['End'] = csv_bed_data['Position'] + 23
    csv_bed_data.rename(columns={'Position': 'Start','chrinfo_extracted':'Chr'}, inplace=True)
    csv_bed_data['Index'] = csv_bed_data.index
    if if_strand:
        csv_bed_data['Score'] = 0
        csv_bed_data = csv_bed_data[ordered_columns]
    return csv_bed_data


''' args:
1. path for bed file
2. chrom type
return a name for the bed file with the type of exprimient and identifier for this label'''
def get_exp_name_to_column(bed_path,chrom_type):
    # get exp name
    exp_name = bed_path.split('/')[-1].split('.')[0]
    # get type of exp - folder after chrom type
    path_parts = bed_path.split('/')
    # Find the index of 'chrom_type'
    openchrom_index = path_parts.index(chrom_type)
    # Extract the folder name after 'chrom_type'
    exp_type = path_parts[openchrom_index + 1]
    column = chrom_type + "_" + exp_type + "_" + exp_name
    return column



'''manual update:
function update a folder of data files that already been marked with chrom information.
args: 1. folder of files needed to be update
2. path for chrom information.
the dir name of the bed files represents the chrom type infromation.'''
def update_info(chrom_tagged_folder,chrom_info_bed):
    # get chrom type from the bed folder name.
    chrom_type =  os.path.basename(chrom_info_bed)
    # get list of the bed files
    bed_files = get_bed_files(chrom_info_bed)
    # iterate on taged folder and retrive data sets from it
    for taged_file in os.listdir(chrom_tagged_folder):
        taged_path = os.path.join(chrom_tagged_folder,taged_file)
        # get a list of columns with the same chrom_type
        columns_list = list_of_bed_columns(chrom_type,taged_path)
        # comprenhense a new bed_files list without columns that already exists in the data
        filtered_bed_files = [bed_file for bed_file in bed_files if not any(col + '.bed' in bed_file for col in columns_list)]
        # apply to the data points the new bed infromation
        for bed_path in filtered_bed_files:
            add_chrom_label(taged_path,bed_path,taged_path,chrom_type)

def list_of_bed_columns(chrom_type,data_path):
    data = pd.read_csv(data_path,sep=",",encoding='latin-1',on_bad_lines='skip')
    chrom_type = chrom_type + "_"
    # Filter columns that start with "chrom type _"
    filtered_columns = [col for col in data.columns if col.startswith(chrom_type)]
    # Extract what comes after "chrom_type"
    bed_names_and_exp = [col.split(chrom_type)[1] for col in filtered_columns]
    # Remove expiriment name
    bed_names = [item.split('_', 1)[1] for item in bed_names_and_exp]
    # return the bed names by column
    return bed_names
def pie_plot_intersection(guideseq40,guideseq50,genome_folder):
    # merge all data into one data frame- get a list of one 40\50 guideseq
    file_paths = create_path_list(guideseq40) + create_path_list(guideseq50)
    list_of_colums = [("Strand",1),('Label_negative',2)] # merge_files get list of tuples
    merge_data = merge_files(file_paths,list_of_colums)
    mergedata_bed_df = create_from_data_bed_data_frame(merge_data,False) # create bed file data
    mergedata_bed = pybedtools.BedTool.from_dataframe(mergedata_bed_df)
    total_intervals = len(mergedata_bed_df) + 1 # include raw 0 
    # create empty list for tuples : (genome_category,intersection_fraction)
    fraction_list = []
    # create genome bed file list
    bed_path_list = get_bed_files(genome_folder)
    # run intersection on each file:
    for path in bed_path_list:
        #check_intervals(path)
        genome_data = pybedtools.BedTool(path) # bed information
        intersection = mergedata_bed.intersect(genome_data,s=False) 
        intersection_amount = count_intervals_bed_file(intersection.fn) # get intersection amount
        fraction = intersection_amount / total_intervals
        genome_category = os.path.basename(path).rstrip('.bed') # exons\introns\etc..
        fraction_list.append((genome_category,fraction))
    sum = 0
    for cat,frac in fraction_list:
        sum = sum + frac
    print(sum)

'''function check for overlaps in off targets locations via data'''
def nagative_intersection(guideseq40,guideseq50):
    # merge all data into one data frame- get a list of one 40\50 guideseq
    file_paths = create_path_list(guideseq40) + create_path_list(guideseq50)
    total =0
    for i,path in enumerate(file_paths):
        print(f"ith iteration: {i}")
        file_df = pd.read_csv(path)
        file_bed_df = create_from_data_bed_data_frame(file_df,False)
        file_bed = pybedtools.BedTool.from_dataframe(file_bed_df)
        for other_path in (file_paths[i+1:]):
            other_df = pd.read_csv(other_path)
            other_bed_df = create_from_data_bed_data_frame(other_df,False)
            other_bed = pybedtools.BedTool.from_dataframe(other_bed_df)
            intersection = file_bed.intersect(other_bed,s=False) 
            intersection_amount = count_intervals_bed_file(intersection.fn)
            total = total+intersection_amount
    print(total)


    # get fraction of intersection and create pie plot
    pass
def create_path_list(combined_folder):
    path_list = []
    for combined_file in os.listdir(combined_folder):
        combined_path = os.path.join(combined_folder,combined_file)
        path_list.append(combined_path)
    return path_list
def check_intervals(genome_data_path):
    
    bed_data = pybedtools.BedTool(genome_data_path)
    total_amount = count_intervals_bed_file(genome_data_path)
   
    intersection_s = bed_data.intersect(bed_data,s=True)
    intersection_no_s = bed_data.intersect(bed_data,s=False)
    amount_intersection_s = count_intervals_bed_file(intersection_s.fn)
    amount_intersection_no_s = count_intervals_bed_file(intersection_no_s.fn)
    if amount_intersection_s > total_amount:
        print("overlapping with Strand")
    if amount_intersection_no_s > total_amount:
        print("overlapping with no Strand")    
def count_intervals_bed_file(file_path):
    command = f"wc -l < {file_path}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return int(result.stdout.strip()) 
   
'''args: 1 path to bedfiles
2 path to guideseq folder'''
if __name__ == "__main__":
    ## the update manualy is dismissed- running manual unfold the '#'
    # input = input("Do you want to update manually an exisintg tagged file? (y/n)")
    # if input == "y":
    #     update_info("/home/alon/masterfiles/guideseq50files/guideseq/0915/chrom_info_tag","/home/alon/masterfiles/guideseq40files/bedfiles/Openchrom")
    #     exit(0)
    #run_chrom_labeling(sys.argv[1],sys.argv[2])
    #pie_plot_intersection("/home/alon/masterfiles/guideseq50files/guideseq/0915params/combined_output","/home/alon/masterfiles/guideseq40files/guideseq/0915params/combined_output","/home/alon/masterfiles/guideseq40files/bedfiles/Genome_info")
    nagative_intersection("/home/alon/masterfiles/guideseq50files/guideseq/0915params/combined_output","/home/alon/masterfiles/guideseq40files/guideseq/0915params/combined_output")
