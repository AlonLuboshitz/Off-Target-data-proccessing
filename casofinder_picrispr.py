import pybedtools
import pandas as pd
import subprocess
import os
'''function creates bed columns from casofiner-output file'''
def create_bed_df_from_casofinder(file_path):
    # casofinder file columns are: seq,chr,position,off-target,strand,missmatches
    columns = [0,"Chr","Start","Off","Strand","Missmatch"] # set columns
    data = pd.read_csv(file_path,delimiter="\t",names=columns)
    data = data[["Chr","Start","Strand","Missmatch"]] # set the data only to this information
    data["End"] = data["Start"] + 23 # add end position to the off-target 23 bp more
    data["Index"] = data.index # set index column
    data["Score"] = 0 # set score column
    ordered_columns = ['Chr', 'Start', 'End', 'Index', 'Score', 'Strand','Missmatch'] # as columns order should be
    data = data[ordered_columns] # set this order
    return data
def delete_intersecting_rows(file1,file2):
    # both files are bed_df_from_casofiner
    file1_bed = pybedtools.BedTool.from_dataframe(file1)
    file2_bed = pybedtools.BedTool.from_dataframe(file2)
    # run intersection, s-strand,u-no duplicates, f=1 complete overlap
    intersection = file2_bed.intersect(file1_bed,s=True,u=True,f=1) # retain indexes the the second file
    intersect_amount = count_intervals_bed_file(intersection.fn)
    print("\nIntersection DataFrame:")
    intersection_df = pd.read_table(intersection.fn, header=None, names=file2.columns)
    print(intersection_df.columns)
    print('intersection 10 lines:\n',intersection_df.head(10))
    overlap_indexes = intersection_df["Index"].values
    #file2.set_index('Index', inplace=True) # set the index column to index
    before_amount = len(file2)
    print('10 lines file 2:\n',file2.head(10))
    file2 = file2.drop(overlap_indexes,axis=0)
    after_amount = len(file2)
    print(f"before: {before_amount}, after: {after_amount}, overlaps: {intersect_amount}")
    if not after_amount == before_amount - intersect_amount:
        print(f"error the before amount {before_amount} - {intersect_amount} isnt {after_amount}")
    return intersect_amount
def delete_by_data(file1,file2):
    # casofinder file columns are: seq,chr,position,off-target,strand,missmatches
    columns = [0,"Chr","Start","Off","Strand","Missmatch"] # set columns
    file1 = pd.read_csv(file1,delimiter="\t",names=columns)
    file2 = pd.read_csv(file2,delimiter="\t",names=columns)
    print(file1.head(5))
    print(file2.head(5))
    # Concatenate DataFrames vertically
    merged = pd.concat([file1, file2], axis=0)
    print(merged.columns)
    print(merged.head(5))
    merged_chr_loc = merged.drop_duplicates(subset=["Chr","Start","Strand"])
    merged_miss = merged.drop_duplicates(subset=["Chr","Start","Strand","Missmatch"])
    merged_off = merged.drop_duplicates(subset=["Chr","Start","Strand","Off"])
    amount = len(merged)
    print(f"amount of data points: {amount}")
    print(f"duplicates by chr and position: {amount-len(merged_chr_loc)}\nduplicates by chr, position, missmatch amount: {amount-len(merged_miss)}\nduplicates by chr, position, OT-seq: {amount-len(merged_off)}")
    inter_sect = delete_duplicates_casofinder(file1,file2)
    if not inter_sect == len(merged) - len(merged_chr_loc):
        print("error")
#    Resetting index if needed
    merged.reset_index(drop=True, inplace=True)
    print(merged.head(5))
def count_intervals_bed_file(file_path):
    command = f"wc -l < {file_path}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return int(result.stdout.strip()) 
def delete_duplicates_casofinder(a,b):
    a = create_bed_df_from_casofinder(a)
    b = create_bed_df_from_casofinder(b)
    delete_intersecting_rows(a,b)
def check_gap(pi_path,casofinder_folder):
    columns = ["Guide","Chr","Position","Off","Strand","Missmatch"] # set columns
    casofinder_data = pd.DataFrame(columns=columns)
    for file in os.listdir(casofinder_folder):
        data = pd.read_csv(os.path.join(casofinder_folder,file),delimiter="\t",names=columns)
        casofinder_data = pd.concat([casofinder_data, data], axis=0)
    casofinder_data.reset_index(drop=True,inplace=True)
    casofinder_data = casofinder_data[["Chr","Position","Strand","Missmatch"]]
    before = len(casofinder_data)
    casofinder_data = casofinder_data.drop_duplicates(subset=["Chr", "Position", "Strand", "Missmatch"])
    after = len(casofinder_data)
    print(f"amount of duplicats: {before-after}")
    # Keep rows where 'Missmatch' is less than 6
    casofinder_data = casofinder_data[casofinder_data['Missmatch'] <= 5]
    after2 = len(casofinder_data) 
    print(f"amount of deletions: {after-after2}, left with {after2} data points")
    
    pidata= pd.read_csv(pi_path)
    pibefore = len(pidata)
    other = pidata.drop_duplicates(subset=["target_chr","target_start","target_strand","mismatch_num","measured"])
    piafter = len(other)
    duplicates = pibefore-piafter
    print(f"amount of duplicats: {pibefore-piafter}")
    pidata = pidata[["target_chr","target_start","target_strand","mismatch_num"]]
    pidata=pidata.rename(columns={"target_chr":"Chr","target_start":"Position"
                                  ,"target_strand":"Strand","mismatch_num":"Missmatch"})
    print(pidata.columns)
    print(pidata.head(5))

    
    merged_df = pd.concat([casofinder_data,pidata],axis=0)
    merged_df.reset_index(drop=True,inplace=True)
    before_merged=len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["Chr", "Position", "Strand", "Missmatch"])
    after_merged = len(merged_df)
    overlaps = before_merged - after_merged - duplicates
    
    counts = merged_df['_merge'].value_counts()
    # Print the counts
    print("Left Only:", counts.get('left_only', 0))
    print("Right Only:", counts.get('right_only', 0))
    print("Both:", counts.get('both', 0))
    
    result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
    print(f'overlaps: {after-len(result_df)}')
    

if __name__ == "__main__":
    check_gap("/home/alon/masterfiles/offtarget_260520_nuc.csv","/home/alon/masterfiles/pythonscripts/casoffinder_outputs_crisprSQL")

        