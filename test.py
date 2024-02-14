guides_lists = [
    ['GAGCAGGGCTGGGGAGAAGGNGG']
,['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']
,['GAAGGCTGAGATCCTGGAGGNGG', 'GGGGGGTTCCAGGGCCTGTCNGG', 'GCACGTGGCCCAGCCTGCTGNGG', 'GAAGGTGGCGTTGTCCCCTTNGG', 'GATTTCTATGACCTGTATGGNGG', 'GCCCTGCTCGTGGTGACCGANGG', 'GTCTCCCTGATCCATCCAGTNGG', 'GAGCCACATTAACCGGCCCTNGG', 'GGAAACTTGGCCACTCTATGNGG', 'GGCCCAGCCTGCTGTGGTACNGG', 'GACATTAAAGATAGTCATCTNGG', 'GAAGCATGACGGACAAGTACNGG', 'GGATTTCCTCCTCGACCACCNGG', 'GGGGCAGCTCCGGCGCTCCTNGG']
,['GACACCTTCTTCCCCAGCCCNGG', 'GCTGGTACACGGCAGGGTCANGG', 'GTCCCTAGTGGCCCCACTGTNGG', 'GGACTGAGGGCCATGGACACNGG', 'GGTGGATGATGGTGCCGTCGNGG', 'GGGATCAGGTGACCCATATTNGG', 'GTCACCAATCCTGTCCCTAGNGG', 'GCTGCAGAAACAGCAAGCCCNGG', 'GGCAGAAACCCTGGTGGTCGNGG', 'GGCCACGGAGCGAGACATCTNGG', 'GGCGCCCTGGCCAGTCGTCTNGG', 'GAGGTTCACTTGATTTCCACNGG', 'GTTTGCGACTCTGACAGAGCNGG', 'GGCCGAGATGTCTCGCTCCGNGG', 'GGGTATTATTGATGCTATTCNGG']
,['GGAGAAGGTGGGGGGGTTCCNGG', 'GTCCCCTCCACCCCACAGTGNGG', 'GCTCGGGGACACAGGATCCCNGG', 'GGACAGTAAGAAGGAAAAACNGG', 'GGCCCCACTGTGGGGTGGAGNGG', 'GATGCTATTCAGGATGCAGTNGG', 'GGTACCTATCGATTGTCAGGNGG', 'GATAACTACACCGAGGAAATNGG', 'GCCGTGGCAAACTGGTACTTNGG', 'GCATTTTCTTCACGGAAACANGG', 'GTATGGAAAATGAGAGCTGCNGG', 'GCGTGACTTCCACATGAGCGNGG', 'GGGAACCCAGCGAGTGAAGANGG', 'GGTTTCACCGAGACCTCAGTNGG']
]

# Flatten the list of lists into a single list
all_guides = [guide for sublist in guides_lists for guide in sublist]

# Use a set to get unique guides
unique_guides = set(all_guides)

# Get the total number of unique guides
total_guides = len(unique_guides)

print("Total number of unique guides:", total_guides)
for i in range(len(guides_lists)):
    for j in range(i + 1, len(guides_lists)):
        common_guides = set(guides_lists[i]) & set(guides_lists[j])
        if common_guides:
            print(f"Lists {i + 1} and {j + 1} have common guides:", common_guides)
        else:
            print(f"Lists {i + 1} and {j + 1} have no common guides.")
            
# def are_lists_equal(list1, list2):
#     return set(list1) == set(list2)

# # Example usage:
# list_a =['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']
# list_b = ['GAAGATGATGGAGTAGATGGNGG', 'GTCAGGGTTCTGGATATCTGNGG', 'GGGGCCACTAGGGACAGGATNGG', 'GCTGCCGCCCAGTGGGACTTNGG', 'GCTGTCCTGAAGTGGACATANGG', 'GTGGTACTGGCCAGCAGCCGNGG', 'GCTGACCCCGCTGGGCAGGCNGG', 'GATTTCCTCCTCGACCACCANGG', 'GAGACCCTGCTCAAGGGCCGNGG', 'GAGAATCAAAATCGGTGAATNGG', 'GAGTAGCGCGAGCACAGCTANGG', 'GCTGGCGATGCCTCGGCTGCNGG', 'GGGCAATGGATTGGTCATCCNGG', 'GCTTCGGCAGGCTGACAGCCNGG']

# if are_lists_equal(list_a, list_b):
#     print("The lists have the same strings.")
# else:
#     print("The lists do not have the same strings.")
# import pandas as pd
# def keep_groups(groups, sum , guides, k, name):
#     columns = ["TP","gRNAS","NUM"]
#     auc_table = pd.DataFrame(columns=columns)
#     guides = list(guides)
#     for i in range(k):
#         indices_list = groups[i]
#         tot_sum = 0
#         guide_list = []
#         for idx in indices_list:
#             tot_sum += sum[idx]
#             guide_list.append(guides[idx])

#         auc_table.loc[i] = tot_sum,guide_list,i+1
#     auc_table.sort_values(by="gRNAS")
#     auc_table.to_csv(name)

        

