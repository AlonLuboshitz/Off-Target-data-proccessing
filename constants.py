'''Constants file'''
## GLOBALS
## COLUMNS:
TARGET_COLUMN = "target"
OFFTARGET_COLUMN = "offtarget_sequence"
CHROM_COLUMN = "chrom"
START_COLUMN = "chromStart"
END_COLUMN = "chromEnd"
BINARY_LABEL_COLUMN = "Label"
REGRESSION_LABEL_COLUMN = "Read_count"
FEATURES_COLUMNS = ["Chromstate_H3K27me3_peaks_binary","Chromstate_H3K27ac_peaks_binary",
                    "Chromstate_H3K9ac_peaks_binary","Chromstate_H3K9me3_peaks_binary"
                     ,"Chromstate_H3K36me3_peaks_binary","Chromstate_ATAC-seq_peaks_binary"
                     ,"Chromstate_H3K4me3_peaks_binary","Chromstate_H3K4me1_peaks_binary"]
