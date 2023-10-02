import os
import pandas as pd
from pybedtools import BedTool
import sys

'''function to create a bed peak file from the peak txt file given'''
def convert_into_bed(path):
    # read file with first raw (header = 0) as columns names   
    df = pd.read_csv(path, sep='\t', header=0)
    # Keep only the relevant columns
    df = df[['chr', 'CI_start', 'CI_stop']]
    # Rename columns
    df = df.rename(columns={'chr':'Chr','CI_start': 'Start', 'CI_stop': 'End'})
    # Save DataFrame as BED file
    bed = BedTool.from_dataframe(df)
    # Get file name
    file_name = path.split('/')[-1].replace('.txt','.bed')
    # Save the bed file in the current dir
    currnet_dir = os.path.join(os.path.dirname(path),f'{file_name}')
    bed.saveas(currnet_dir)                                            








if __name__ == "__main__":
    convert_into_bed(sys.argv[1])