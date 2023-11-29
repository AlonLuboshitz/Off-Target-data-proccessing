# code for labeling positive (off target chromosomal positions by guideseq).
import pandas as pd
import os
import re
import sys
from shutil import rmtree

'''given an path for an indentified.txt, filter by bool the BED_site_name culom.
BED_site_name colom if not null represents a recognisble off-target site.
1. filter.
2. add label colum = 1 (positive)
3. get expriment name
4. create csv filtered file in the output path named with expirement name + "_label_sub_only"
columns kept are: Chromosome,start_missmatch_position,missmatches,position,seq,bi(reads),filenme
start_missmatch is corresponding to position in cas-offinder data.'''
def label_identified(input_path,output_path):
    identified_file = pd.read_csv(input_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
    # 1. filter by bed_site and by Sub only seq (varifying there is only subs option)
    filtered_df = identified_file.dropna(subset=['BED_Site_Name','Site_SubstitutionsOnly.Sequence'])
    # 2. filter by missmatch count <=6
    missmatch_condition = filtered_df['Site_SubstitutionsOnly.NumSubstitutions'] <= 6
    # Apply the additional filtering using the loc indexer
    final_filtered_df = filtered_df.loc[missmatch_condition]
    selected_colums = ['WindowChromosome','Site_SubstitutionsOnly.Start','Site_SubstitutionsOnly.NumSubstitutions','Site_SubstitutionsOnly.Sequence','Position','TargetSequence','bi.sum.mi','Filename']
    new_df = pd.DataFrame(data=final_filtered_df,columns=selected_colums)
    new_df.rename(columns={'Site_SubstitutionsOnly.NumSubstitutions': 'Missmatches'}, inplace=True)
    # 2. set positive label
    new_df['Label'] = 1
    # 3. get expirement name
    exp_name = new_df.iloc[0]['Filename']
    exp_name = exp_name.replace('.sam','')
    exp_name = exp_name + "_label_sub_only"
    # 4. get file name and add it to output path 
    output_filename = f"{exp_name}.csv"
    output_path = os.path.join(output_path,output_filename)
    # 5. create csv file
    new_df.to_csv(output_path, index=False)
    print("created {} file in folder: {}".format(output_filename,output_path))
'''
function gets a folder with guide-seq identified txt files.
create a new output folder (if not exists) with csv files filtered by
label identified function
folder name created: _labeled_sub_only'''
def process_folder(input_folder):
    # create folder for the labeled procceses identified.
    label_output_folder = input_folder + '_labeled_sub_only'
    create_folder(label_output_folder)
    cas_offinder_output_folder = input_folder + '_casoffinder_inputs'
    create_folder(cas_offinder_output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(input_folder, filename)
            label_identified(txt_file_path, label_output_folder)
            create_csofinder_input_by_identified(txt_file_path,cas_offinder_output_folder)
'''create folder in spesefic path'''
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("created new folder: ", path)

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
def create_cas_ofinder_inputs(table,target_column,output_name,path_for_casofiner):
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
     

'''
function looks for multiple duplicates from the same exprimenet and merge the data from them
args - folder paths, number of duplicated, file ending - exmp: '_labeled.csv, erase: for erasing the previous folder.'
NOTE: all files should have the same ending, for example: -D(n)_labeled
NOTE: before set should *n_duplicates more then after
function calls each file by iterating the number on duplicates and changing the file ending '''
def merge_positives(folder_path,n_duplicates,file_ending,erase):
    file_names = os.listdir(folder_path)
    # more then 1 duplicate
    if int(n_duplicates) > 1:
        # create pattern of xxx-D(n) + suffix
        pattern = r"(.+?-D)\d+" + re.escape(file_ending)
        # create one string from all the file names and find the matching pattern.
        file_names = ''.join(file_names)
        mathces = re.findall(pattern,file_names)
        # use a set to create a unique value for n duplicates
        unique = list(set(mathces))
        print('before mergning: ',len(mathces))
        print('after mergning: ',len(unique))
    else : 
        unique = file_names
    # get tuple list - df, name
    final_file_list = mergning(unique,n_duplicates,file_ending,folder_path)   
    # create folder for output combined expriment:
    # remove .csv\.txt from ending.
    file_ending = file_ending.rsplit('.', 1)[0]
    # create folder
    output_path = os.path.join(os.path.dirname(folder_path),f'{file_ending}_merged')
    if not os.path.exists(output_path):
        os.mkdir(output_path) 
        print ("create folder for merged files in: {}".format(output_path))
    # create csv file from eached grouped df.
    for tuple in final_file_list:
        name = tuple[1] + '.csv'
        temp_output_path = os.path.join(output_path,name)
        tuple[0].to_csv(temp_output_path,index=False)
    if erase == "True":
        try:
            rmtree(folder_path)
            print(f"Directory '{folder_path}' and its contents have been removed.")
        except Exception as e:
            print(f"Error: {e}") 


'''args: 1. files list
2. n duplicates
3. file ending - e.a - _labeled.csv
4. path for folder with labeled files.
return tuple of merged files - data frame, file name.'''    
def mergning(files,n,file_ending,folder_path):
    final_file_list = []
    # more then 1 duplicate per file
    if int(n) > 1:
        for file_name in files:
            # create n duplicates file list
            n_file_list =[]
            for count in range(int(n)):
                # create string for matching duplicate file
                temp_file_name = file_name + str(count+1) + file_ending 
                input_path = os.path.join(folder_path,f'{temp_file_name}')
                # append df into the list
                n_file_list.append(pd.read_csv(input_path,sep=",",encoding='latin-1',on_bad_lines='skip'))
            # n_file_list have all duplicates in it, merge them:
            merged_df = pd.concat(n_file_list, axis=0, ignore_index=True)
            print ('before grouping: ',len(merged_df))
            # group by position, number of missmatches, and siteseq. sum the bi - reads
            grouped_df = merged_df.groupby(['Site_SubstitutionsOnly.Start','Missmatches','Site_SubstitutionsOnly.Sequence','WindowChromosome','Label','TargetSequence'], as_index=False)['bi.sum.mi'].sum()
            print ('after grouping: ',len(grouped_df))
            # append df to final list
            final_file_list.append((grouped_df,file_name)) 
    # no duplicates, keep original file with less columns       
    else :
        for file_name in files:
            input_path = os.path.join(folder_path,f'{file_name}')
            df = pd.read_csv(input_path,sep=",",encoding='latin-1',on_bad_lines='skip')
            file_name = file_name.replace(".csv","")
            df = df[['Site_SubstitutionsOnly.Start','Missmatches','Site_SubstitutionsOnly.Sequence','WindowChromosome','Label','TargetSequence','bi.sum.mi']]
            final_file_list.append((df,file_name))

            
    return final_file_list
'''
function gets path for identified (guideseq output data) folder and calls:
process_folder function, which creates csv folder named: identified_labeled_sub_only
this folder is then used in merge_positive function to merge the duplicate guide-seq expriments
argv 1 - path to identified.
argv 2 -  number of duplicated expriments
argv 3 - keep the identified label folder or erase it
'''
if __name__ == '__main__':
    # process_folder(sys.argv[1])
    # # join path to identified +  labeled only.
    # labeld_path = sys.argv[1] + f'_labeled_sub_only' 
    # merge_positives(labeld_path,sys.argv[2],'_label_sub_only.csv',sys.argv[3])
    create_cas_ofinder_inputs(table="/home/alon/masterfiles/pythonscripts/Changeseq/GUIDE-seq.xlsx",target_column="target",output_name="Changeseq/Guide_seq_casofinder",path_for_casofiner="hg38")

        


