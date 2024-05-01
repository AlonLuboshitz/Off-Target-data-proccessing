'''Constants file'''
## GLOBALS
## COLUMNS:
TARGET_COLUMN = "target"
OFFTARGET_COLUMN = "offtarget_sequence"
CHROM_COLUMN = "chrom"
START_COLUMN = "chromStart"
END_COLUMN = "chromEnd"
BINARY_LABEL_COLUMN = "Label"
# FEATURES_COLUMNS = ["Chromstate_H3K27me3_peaks_binary","Chromstate_H3K27ac_peaks_binary",
#                     "Chromstate_H3K9ac_peaks_binary","Chromstate_H3K9me3_peaks_binary"
#                     ,"Chromstate_H3K36me3_peaks_binary","Chromstate_atacseq_peaks_binary"
#                     ,"Chromstate_h3k4me3_peaks_binary","Chromstate_H3K4me1_peaks_binary"]

FEATURES_COLUMNS = ["Chromstate_atacseq_peaks_binary"
                    ,"Chromstate_h3k4me3_peaks_binary", 
                    "Chromstate_H3K27ac_peaks_binary"]

## DICTS:
METHOD_DICT = {
            1: "Only sequence",
            2: "Epigenetic by features",
            3: "Base pair epigenetic in Sequence",
            4: "Seperate epigenetics "
        } 

## PATHS:
DATA_PATH = "/home/alon/masterfiles/pythonscripts/Data"


## CHANGESEQ
    # Epigenetics
EPIGENETIC_FOLDER = "/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics"
BED_FILES_FOLDER = "/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/Chromstate"
BIG_WIG_FOLDER = "/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig"
    # Preprocessed data
POSITIVE_PATH = "/home/alon/masterfiles/pythonscripts/Data/Raw_data/GUIDE-seq.xlsx"
NEGATIVE_PATH = "/home/alon/masterfiles/pythonscripts/Casofinder_outputs/Change_seq/Change_seq_guide_seq_outputs.txt"
    # Processed data
CHANGESEQ_CASO_EPI = "/home/alon/masterfiles/pythonscripts/Data/Processed_data/Change_seq/merged_csgs_casofinder_withEpigenetic.csv"
CHANGESEQ_GS_EPI = "/home/alon/masterfiles/pythonscripts/Data/Processed_data/Change_seq/merged_csgs_withEpigenetic.csv"
MERGED_TEST = "/home/alon/masterfiles/pythonscripts/Changeseq/merged_test.csv"
    # 5k tests
CHANGESEQ_5k_TEST = "/home/alon/masterfiles/pythonscripts/Data/Test_guides/Change_seq/Test_5K_guides.csv"
