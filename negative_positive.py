# NEGATIVE - POSITIVE
# Functions: 1. negative-positive  
# 2. negative to csv files.
import pandas as pd
import os
import sys

''' 
function recives two folders: negative - (casoffinder,etc..), positive - guideseq data
args - paths 
creates new "combined" folder in which every guide sequence (=TragetSeq) represnted in a
_combined.csv file with labels - 1/0. 1 for positive (guideseq) 0 - only in cas-offinder 
NOTE: if not all the guide-seq positions are found in the cas-ofinder data
the combined output txt is mark as "_combined_error.csv" '''
def negative_positive(positive_folder,negative_folder):
    # create folder for calculated files.
    combined_path = os.path.dirname(positive_folder) + f'/combined_output'
    if not os.path.exists(combined_path):
        os.mkdir(combined_path)
    # iterate on positive files
    positive_files = os.listdir(positive_folder)
    for positive_file in positive_files:
        positive_path = os.path.join(positive_folder,positive_file)    
        # 1. get target seq from positive file and look for it corresponding file in the 
        # negative file folder.
        positive_file = pd.read_csv(positive_path,sep=",",encoding='latin-1',on_bad_lines='skip')
        positive_seq = positive_file.at[0,'TargetSequence']
        # concanicate '/seq_output.csv'
        output_filename = f"{positive_seq}_output.csv"
        negative_file_path = os.path.join(negative_folder,output_filename)
        # if file not exists continue to next one!
        try:
            negative_file = pd.read_csv(negative_file_path,sep=",",encoding='latin-1',on_bad_lines='skip')
        except FileNotFoundError as e:
            print('File Not found error: ',e)
            continue

        # 2. validate target seq equals in both files.
        negative_seq = negative_file.at[0,'TargetSequence']
        if positive_seq == negative_seq:
            # extract chrT where T- string after chr from the chrinfo column in cas-offinder output
            negative_file['chrinfo_extracted'] = negative_file['Chrinfo'].str.extract(r'(chr[^\s]+)')
            # filter equalaty of correspoding poistion in cas-offinder and guideseq
            # validate its the same chr aswell!
            filtered_positions = negative_file[negative_file['Position'].isin(positive_file['Site_SubstitutionsOnly.Start'])
                                               & negative_file['chrinfo_extracted'].isin(positive_file['WindowChromosome'])]
            # set label for negatives
            negative_file['Label'] = 0
            # merge data from CO and guideseq
            merged_negative = negative_file.merge(positive_file, how='left',
                                      left_on=['chrinfo_extracted', 'Position'],
                                      right_on=['WindowChromosome', 'Site_SubstitutionsOnly.Start'],
                                      suffixes=('_negative', '_positive'))
            
            # Mark the 'Label_negative' column as 1 for the filtered positions
            merged_negative.loc[filtered_positions.index, 'Label_negative'] = 1
            merged_negative = merged_negative.sort_values(by='Label_positive', ascending=False)
            merged_negative = merged_negative.sort_values(by='bi.sum.mi', ascending=False)
            # not all guideseq missmatched were found in the cas-offinder
            # create the file with _errored.csv
            error_str = f'{positive_seq}_combined.csv'
            if not len(positive_file)==len(filtered_positions):
                error_str = f'{positive_seq}_combined_errored.csv'
            temp_output = os.path.join(combined_path,error_str)
            merged_negative.to_csv(temp_output,index=False)
          

                

'''
create csv files with corresponding header - [targetseq,chrinfo,position,siteseq,strand,missmatch]
args - getting path to a folder containing output.txt files from cas-offinder.
''' 
def negatives_tocsv(file_path):
    # create output csv folder for cas-offinder outputs.txt
    outputpath = os.path.dirname(file_path)
    outputpath = outputpath + f'/casoffinder_outputs_csvs'
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        # iterate on outputs.txt files and transform into csv with headers.
        files = os.listdir(file_path)
        for file in files:
            # replace .txt with .csv
            file_ends = file.replace('.txt','.csv')
            # read .txt file and create new df with headers
            file_temp_path = os.path.join(file_path,file)
            try:
                negative_file = pd.read_csv(file_temp_path,sep="\t",encoding='latin-1',on_bad_lines='skip')
            except pd.errors.EmptyDataError as e:
                print(file," is empty, continuing to next file!")
                continue
            columns = ['TargetSequence','Chrinfo','Position','Siteseq','Strand','Missmatches']
            negative_file.columns = columns
            # create output path in the new created folder and the csv file in it.
            temp_outputpath = os.path.join(outputpath,file_ends)
            negative_file.to_csv(temp_outputpath, index=False)
    return
'''
main function gets two args:
1 - path to outputs folder of casoffinder - uses negative_tocsv function to 
create csv corresponding folder named: casoffinder_outputs_csvs
2 - path to labeled guide seq files.
runs the output folder of negative_tocsv with negative_positive to create combined files. '''
if __name__ == '__main__':
    # create outputs csv folder
    negatives_tocsv(sys.argv[1])
    # reatian path to parent folder 
    path_to_csv = os.path.join(os.path.dirname(sys.argv[1]),f'casoffinder_outputs_csvs')
    negative_positive(sys.argv[2],path_to_csv)
         

