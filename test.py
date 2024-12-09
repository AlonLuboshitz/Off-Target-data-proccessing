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
import os
import shutil

# Define the source and destination base directories
source_base = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/7_partition_50/binary"
dest_base = "/localdata/alon/ML_results/Change-seq/vivo-vitro/Classification/CNN/Ensemble/Epigenetics_by_features/7_partition/1_ensembels/50_models/binary"

# Iterate through all folders in the source binary directory
for folder in os.listdir(source_base):
    source_scores_dir = os.path.join(source_base,f'{folder}/Scores' )
    dest_scores_dir = os.path.join(dest_base, f'{folder}/Scores')
    
    # Check if the source Scores directory exists
    if os.path.exists(source_scores_dir):
        source_file = os.path.join(source_scores_dir, "ensmbel_1.csv")
        # Check if the file exists before copying
        if os.path.exists(source_file):
            # Create the destination Scores directory if it doesn't exist
            os.makedirs(dest_scores_dir, exist_ok=True)
            # Copy the file to the destination directory
            shutil.copy(source_file, dest_scores_dir)
            print(f"Copied {source_file} to {dest_scores_dir}")
        else:
            print(f"File not found: {source_file}")
    else:
        print(f"Scores directory not found in: {os.path.join(source_base, folder)}")
