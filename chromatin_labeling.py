# script for getting a bed file with chromatin info and cas-offinder output.csv
# creating a label for the bed file expriment with 1 if intersects with the cas-offinder or 0 if not.
import os
import pandas as pd
import pybedtools
import sys
'''function gets path for combined file - with active\inactive labeling
bed_path - bedfile for chromatin info e.g - accesability, methylation, phopholyration etc..
output_path - path to place new file with intersections with the bed file information.
chrom_info - folder name of type of chrom info - accsesbility, mythlation, phsos...
'''
def add_opencrhom_label(combined_path,bed_path,output_path,chrom_info):
    # check if already the combined file has been labeled with difrrenet bed file:
    # retrive ending of file name 
    file_path_ending =  combined_path.split('/')[-1].split('.')[0] + "_" + chrom_info + ".csv"
    # merge the ending with the output folder
    output_path = os.path.join(output_path,file_path_ending)
    # if the file exists -> use this file instead of the original combined.
    if os.path.exists(output_path):
        combined_path = output_path
        
    # Read BED file into a DataFrame bed file has no headers
    bed_data = pd.read_csv(bed_path, sep='\t', header=None)
    #get number of columns
    num_columns = bed_data.shape[1]
    # Create column names based on the number of columns
    column_names = ['Chr', 'Start', 'End'] + [str(i) for i in range(4, num_columns + 1)]
    # Assign the new column names to the DataFrame
    bed_data.columns = column_names 
    # get combined file to df.
    combined_data = pd.read_csv(combined_path)
    # convert combined df into bedfile with : Chr,Start,End,Index 
    csv_bed_data = combined_data[['chrinfo_extracted', 'Position']]
    csv_bed_data['End'] = csv_bed_data['Position'] + 1 
    csv_bed_data.rename(columns={'Position': 'Start','chrinfo_extracted':'Chr'}, inplace=True)
    csv_bed_data['Index'] = csv_bed_data.index
    csv_bed = pybedtools.BedTool.from_dataframe(csv_bed_data)
    # Perform the intersection using pybedtools
    intersections = csv_bed.intersect(bed_path, u=True, s=False)
    # convert intersection output into df.
    intersection_df = intersections.to_dataframe(header=None, names=['Chr', 'Start','End','Index'])
    # Extract the filename without extension from the absolute path and add user-input
    column_bed_filename = bed_path.split('/')[-1].split('.')[0]
    column_bed_filename = chrom_info + "_" + column_bed_filename
    # Add a new column to combined_data based on the intersection result - 1 for intersect
    combined_data[column_bed_filename] = 0  
    try:
        combined_data.loc[intersection_df['Index'], column_bed_filename] = 1
    except KeyError as e:
        print(combined_path,': file has no intersections output will be with 0')
        input("press anything to continue: ")
    # clean data
    combined_data.to_csv(output_path, index=False)
'''function gets expirement data frame and cleans it from uneccesry data for further applications'''
# def clean_data()
'''folder of combined_files - e.a labeled.
iterated on files and pass each one to opencrhom function
output path is given by user via stdin
chrom_info - name of folder with the bed files. e.a folder name openchrom consists bed files of
chrom accesasbility.'''
def apply_for_folder(folder_combined_path,bed_path,chrom_info):
    # create a new folder named after that chrom info.
    output_path = os.path.join(os.path.dirname(folder_combined_path),chrom_info)
    if not os.path.exists(output_path):
        os.mkdir(output_path) 
    # iterate combined files
    print ("labeling combined_output folder {}, with bed file: {}, from type: {}".format(folder_combined_path,bed_path,chrom_info))
    for combined_file in os.listdir(folder_combined_path):
        combined_path = os.path.join(folder_combined_path,combined_file)
        # label the combined file vie bed fils
        add_opencrhom_label(combined_path,bed_path,output_path,chrom_info) 
''' function gets 3 arguments:
1 - path for folder with combined paths.
2 - path for a bed file.
3 - type of bed information - openchrom\mtyhl etc..'''
if __name__ == "__main__":
    apply_for_folder(sys.argv[1],sys.argv[2],sys.argv[3])


# apply_for_folder('/home/alon/masterfiles/guideseq40files/guideseq/0915params/combined_output','/home/alon/masterfiles/guideseq40files/bedfiles/Openchrom/Atacdb/ATAC_1014.bed','Openchrom')
