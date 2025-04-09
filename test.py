# guides_lists = [
#     ['GAGCAGGGCTGGGGAGAAGGNGG']
# ,['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']
# ,['GAAGGCTGAGATCCTGGAGGNGG', 'GGGGGGTTCCAGGGCCTGTCNGG', 'GCACGTGGCCCAGCCTGCTGNGG', 'GAAGGTGGCGTTGTCCCCTTNGG', 'GATTTCTATGACCTGTATGGNGG', 'GCCCTGCTCGTGGTGACCGANGG', 'GTCTCCCTGATCCATCCAGTNGG', 'GAGCCACATTAACCGGCCCTNGG', 'GGAAACTTGGCCACTCTATGNGG', 'GGCCCAGCCTGCTGTGGTACNGG', 'GACATTAAAGATAGTCATCTNGG', 'GAAGCATGACGGACAAGTACNGG', 'GGATTTCCTCCTCGACCACCNGG', 'GGGGCAGCTCCGGCGCTCCTNGG']
# ,['GACACCTTCTTCCCCAGCCCNGG', 'GCTGGTACACGGCAGGGTCANGG', 'GTCCCTAGTGGCCCCACTGTNGG', 'GGACTGAGGGCCATGGACACNGG', 'GGTGGATGATGGTGCCGTCGNGG', 'GGGATCAGGTGACCCATATTNGG', 'GTCACCAATCCTGTCCCTAGNGG', 'GCTGCAGAAACAGCAAGCCCNGG', 'GGCAGAAACCCTGGTGGTCGNGG', 'GGCCACGGAGCGAGACATCTNGG', 'GGCGCCCTGGCCAGTCGTCTNGG', 'GAGGTTCACTTGATTTCCACNGG', 'GTTTGCGACTCTGACAGAGCNGG', 'GGCCGAGATGTCTCGCTCCGNGG', 'GGGTATTATTGATGCTATTCNGG']
# ,['GGAGAAGGTGGGGGGGTTCCNGG', 'GTCCCCTCCACCCCACAGTGNGG', 'GCTCGGGGACACAGGATCCCNGG', 'GGACAGTAAGAAGGAAAAACNGG', 'GGCCCCACTGTGGGGTGGAGNGG', 'GATGCTATTCAGGATGCAGTNGG', 'GGTACCTATCGATTGTCAGGNGG', 'GATAACTACACCGAGGAAATNGG', 'GCCGTGGCAAACTGGTACTTNGG', 'GCATTTTCTTCACGGAAACANGG', 'GTATGGAAAATGAGAGCTGCNGG', 'GCGTGACTTCCACATGAGCGNGG', 'GGGAACCCAGCGAGTGAAGANGG', 'GGTTTCACCGAGACCTCAGTNGG']
# ]

# # Flatten the list of lists into a single list
# all_guides = [guide for sublist in guides_lists for guide in sublist]

# # Use a set to get unique guides
# unique_guides = set(all_guides)

# # Get the total number of unique guides
# total_guides = len(unique_guides)

# print("Total number of unique guides:", total_guides)
# for i in range(len(guides_lists)):
#     for j in range(i + 1, len(guides_lists)):
#         common_guides = set(guides_lists[i]) & set(guides_lists[j])
#         if common_guides:
#             print(f"Lists {i + 1} and {j + 1} have common guides:", common_guides)
#         else:
#             print(f"Lists {i + 1} and {j + 1} have no common guides.")
            
# def are_lists_equal(list1, list2):
#     return set(list1) == set(list2)

# # Example usage:
# list_a =['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']
# list_b = ['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']

# if are_lists_equal(list_a, list_b):
#     print("The lists have the same strings.")
# else:
#     print("The lists do not have the same strings.")


# def count_sgrnas_in_rows(filename):
#     with open(filename, 'r') as file:
#         for idx, line in enumerate(file):
#             # Split each row by commas to get individual sgRNAs
#             sgrnas = line.strip().split(',')
#             # Count the number of sgRNAs in this row
#             sgrna_count = len(sgrnas)
#             print(f"Row {idx + 1}: {sgrna_count} sgRNAs")

# count_sgrnas_in_rows("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Changeseq/5k_ensmbels_change.txt")
# from features_engineering import generate_features_and_labels
# from multi_process_utilities import *
# import numpy as np

# Columns_dict = {
#     "TARGET_COLUMN": "target",
#     "REALIGNED_COLUMN": "realigned_target",
#     "OFFTARGET_COLUMN": "offtarget_sequence",
#     "CHROM_COLUMN": "chrom",
#     "START_COLUMN": "chromStart",
#     "END_COLUMN": "chromEnd",
#     "Y_LABEL_COLUMN": "Label",
#     "REGRESSION_LABEL_COLUMN": "Read_count"
# }
# x_,y_,g = generate_features_and_labels(data_path="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/Processed_data/vivo-vitro-78_withEpigenetic.csv",
#                                      manager=None,if_bp=False,if_only_seq=True,if_seperate_epi=False,epigenetic_window_size=20,features_columns=None,
#                                      if_data_reproducibility=False,columns_dict=Columns_dict,sequence_coding_type=2,if_bulges=True)
# shared_x = convert_x_feature_list_to_shared_object(x_)
# shared_y = convert_y_label_list_to_shared_object(y_,task="classification")

#     # Store the original shapes for later reshaping
# print('x:\n',x_[0][:20])
# print('y\n',y_[0][:20])
# x__ = convert_shared_x_to_x_feature_list(shared_x)
# y__ = convert_shared_y_to_y_label_list(shared_y)
# print('x:\n',x__[0][:20])
# print('y\n',y__[0][:20])
# from utilities import extract_scores_labels_indexes_from_files
# from features_engineering import keep_indexes_per_guide
# from evaluation import keep_indexes_from_scores_labels_indexes, evaluate_classification

# def validate_equality_between_scores(fn_scores,fn_labels,fn_indexes,sci_score,sci_labels,sci_indexes):
#     '''Validate that two sets of scores, labels and indexes are equal
#     and equal in their evaluations.'''
#     f5_predictions, f5_labels, f5_indexes = fn_scores, fn_labels, fn_indexes
#     sci_predictions, sci_labels, sci_indexes = sci_score, sci_labels, sci_indexes
#     print('equlaity of scores by guides',np.array_equal(f5_predictions,sci_predictions))
    
#     print('equlaity of labels by guides',np.array_equal(f5_labels,sci_labels))
#     print('equlaity of indexes by guides',np.array_equal(f5_indexes,sci_indexes))
#     results_f5, rates_dict_f5 = evaluate_classification(f5_labels, f5_predictions, return_rates=True)
#     auroc_f5, auprc_f5, n_rank_f5, last_fn_index_f5, last_fn_ratio_f5 = results_f5
#     fpr_f5, tpr_f5, precision_f5, recall_f5 = rates_dict_f5.values()
#     results_sci, rates_dict_sci = evaluate_classification(sci_labels, sci_predictions, return_rates=True)
#     auroc_sci, auprc_sci, n_rank_sci, last_fn_index_sci, last_fn_ratio_sci = results_sci
#     fpr_sci, tpr_sci, precision_sci, recall_sci = rates_dict_sci.values()
#     print('equality of auroc',auroc_f5,auroc_sci,np.array_equal(auroc_f5,auroc_sci))

#     print('equality of auprc',auprc_f5,auprc_sci,np.array_equal(auprc_f5,auprc_sci))
#     print('equality of n_rank',n_rank_f5,n_rank_sci,np.array_equal(n_rank_f5,n_rank_sci))
#     print('equality of last_fn_index',last_fn_index_f5,last_fn_index_sci,np.array_equal(last_fn_index_f5,last_fn_index_sci))   
#     print('equality of last_fn_ratio',last_fn_ratio_f5,last_fn_ratio_sci,np.array_equal(last_fn_ratio_f5,last_fn_ratio_sci))
#     print('equality of fpr',np.array_equal(fpr_f5,fpr_sci))
#     print('equality of tpr',np.array_equal(tpr_f5,tpr_sci))
#     print('equality of precision',np.array_equal(precision_f5,precision_sci))
#     print('equality of recall',np.array_equal(recall_f5,recall_sci))

# def equality_of_two_scitifinc_notations(sci1,sci2):
        
    
#     f5_predictions, f5_labels, f5_indexes = extract_scores_labels_indexes_from_files([sci1])
#     sci_predictions, sci_labels, sci_indexes = extract_scores_labels_indexes_from_files([sci2])
#     print('equality of arrays of score arrays: ',np.array_equal(f5_predictions,sci_predictions))
#     f5_predictions = np.mean(f5_predictions,axis=0)
#     sci_predictions = np.mean(sci_predictions,axis=0)
#     print('equlaity of mean arrays: ',np.array_equal(f5_predictions,sci_predictions))
#     vals,counts = np.unique(f5_predictions, return_counts=True)
#     print(f'{len(counts)} of distinct values in notation 1 {sci1}')
#     vals,counts = np.unique(sci_predictions, return_counts=True)
#     print(f'{len(counts)} of distinct values in notation 2 {sci2}')
#     lazaroto = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/TrueOT/Refined_TrueOT_Lazzarotto_withEpigenetic.csv"
#     guide_indexes = keep_indexes_per_guide(lazaroto,'target')
#     print("all guides")
#     validate_equality_between_scores(f5_predictions,f5_labels,f5_indexes,sci_predictions,sci_labels,sci_indexes)
#     for guide, indexes in guide_indexes.items():
#         f5_predictions_i, f5_labels_i, f5_indexes_i = keep_indexes_from_scores_labels_indexes(f5_predictions, f5_labels, f5_indexes, indexes)
#         sci_predictions_i, sci_labels_i, sci_indexes_i = keep_indexes_from_scores_labels_indexes(sci_predictions, sci_labels, sci_indexes, indexes)
#         print("checking guide:", guide)
#         validate_equality_between_scores(f5_predictions_i,f5_labels_i,f5_indexes_i,sci_predictions_i,sci_labels_i,sci_indexes_i)
# f5_scores = "/home/dsi/lubosha/score_comparison/ensemble_1.csv"
# sci_scores = "/home/dsi/lubosha/score_comparison/ensemble_1_e.csv"
# equality_of_two_scitifinc_notations(f5_scores,sci_scores)
import os
import re
import pandas as pd
            
def write_to_files(f_sites,f_numbers, rhamp_seq_sites, panel_sites, guide_name,len_merged_data):
    
    common_sites = rhamp_seq_sites.intersection(panel_sites)
    missing_sites = rhamp_seq_sites.difference(panel_sites)
    f_numbers.write(f"Panel {guide_name} has {len(rhamp_seq_sites)} sites in rhampseq results and {len(panel_sites)} sites in panel\n")
    f_numbers.write(f'Common sites: {len(common_sites)}, Missing sites: {len(missing_sites)}')
    if len_merged_data != len(common_sites):
        f_numbers.write(f"\nNumber of common sites is not equal to the number of merged data rows")
    f_numbers.write('\n\n')
    f_sites.write(f"Panel: {guide_name}\nCommon sites: {common_sites}\nMissing sites: {missing_sites}\n\n")

rhamp_seq_with_coords_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/rHamp-seqs_crispector/With_coords'
rhamp_seq_results_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/rHamp-seqs_crispector/Merged'
panels_path = '/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/rHamp-seqs_crispector/Merged/ALL_rhAmpSeq_panels'
panel_files = os.listdir(panels_path)
rhamp_seq_results_files = os.listdir(rhamp_seq_results_path)
rhamp_seq_results_files = [file for file in rhamp_seq_results_files if re.match(r'^.*\.csv$', file)]
#extract the panel name from the file name
pattern = r"(.+?)-D"
dict_panels = {}
for file in rhamp_seq_results_files:
    panel = re.search(pattern,file).group(1)
    # find Final.txt file for each panel
    temp_panel_path = os.path.join(panels_path,panel)
    temp_panel_files = os.listdir(temp_panel_path)
    final_file = [file for file in temp_panel_files if re.match(r'^.*final\.xlsx$', file.lower())]
    if final_file:
        final_file = final_file[0]
        dict_panels[panel] = (os.path.join(rhamp_seq_results_path,file),os.path.join(temp_panel_path,final_file))
    else:
        print(f"Final file not found for panel {panel}")
    if panel == "RAG2":
        dict_panels[panel] = (os.path.join(rhamp_seq_results_path,file),os.path.join(temp_panel_path,"RAG2_full_rhAmpSeq_panel_data.csv"))
print(dict_panels)
matching_sites = os.path.join(rhamp_seq_results_path,'matching_sites.txt')
matching_numbers= os.path.join(rhamp_seq_results_path,'matching_numbers.txt')
with open(matching_sites, 'w') as f_sites, open(matching_numbers, 'w') as f_numbers:
    ## merge rhampseq results with off target coordinates
    xlsx_columns = ['Chrom','chromStart','chromEnd','Site Name','score','Strand','1']
    merged_columns_order = ['Chrom','chromStart','chromEnd','Site Name','Strand','avg_editing']
    
    for guide_name,files_tuple in dict_panels.items():
        if guide_name == "gGM10-FANCA":
            off_target_coordinates = pd.read_excel(files_tuple[1],header=None)
            off_target_coordinates.drop(columns=off_target_coordinates.columns[6], inplace=True)
            off_target_coordinates.columns = xlsx_columns
        
        elif guide_name == "RAG2":
            off_target_coordinates = pd.read_csv(files_tuple[1])
            columns_rename = {'Chromosome': 'Chrom', 'amplicon ID': 'Site Name', 'guide start position': 'chromStart', 'guide end position': 'chromEnd', 'guide strand': 'Strand'}
            off_target_coordinates.rename(columns=columns_rename, inplace=True)
        else:
            off_target_coordinates = pd.read_excel(files_tuple[1],header=None,names=xlsx_columns)
        rhamp_seq_results = pd.read_csv(files_tuple[0])
        rhamp_seq_results['Site Name'] = rhamp_seq_results['Site Name'].str.replace('-','_')
        
        try:
            off_target_coordinates['Site Name'] = off_target_coordinates['Site Name'].astype(str)
            off_target_coordinates['Site Name'] = off_target_coordinates['Site Name'].str.replace('-','_')
        except Exception as e:
            print(e)
            print(f'Error in panel {guide_name}\nPath: {files_tuple[1]}')
            continue
        rhamp_seq_sites = set(rhamp_seq_results.loc[rhamp_seq_results['avg_editing'].notna(), 'Site Name'])
        panel_sites = set(off_target_coordinates['Site Name'])
        
        merged_data = pd.merge(rhamp_seq_results,off_target_coordinates,on='Site Name',how='inner')

        write_to_files(f_sites,f_numbers, rhamp_seq_sites, panel_sites, guide_name,len(merged_data))
        try:

            merged_data = merged_data[merged_columns_order]
        except Exception as e:
            print(e)
            print(f'Error in panel {guide_name}\nPath: {files_tuple[0]}\n{files_tuple[1]}')
            print(merged_data.columns)
            continue
        merged_data.to_csv(os.path.join(rhamp_seq_with_coords_path,f'{guide_name}.csv'),index=False)

