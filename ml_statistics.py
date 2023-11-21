## script for analysis of ml results
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
'''
wilxocon test for paired samples, x-y
recomendation is to give the function only the x-y difrences vector.'''
def get_wilxocon_sign(x,y,test):
    # define test direction
    alternative = get_alternative(test)
    x = x.round(5) # 5 decimels after the dot
    y = y.round(5)
    difrrences = (x-y)
    T_statistic,p_val = wilcoxon(x=difrrences,y=None,alternative=alternative)
    stats_dict = {"T_sts":T_statistic,"P.v":p_val}
    return stats_dict
def get_alternative(test):
    if test == "<":
        alternative = "less"
    elif test == ">":
        alternative = "greater"
    else: alternative = "two-sided"
    return alternative
def extract_ml_data(ml_summary_table):
    # Read all sheets into a dictionary of DataFrames
    all_sheets = pd.read_excel(ml_summary_table, sheet_name=None)
    modified_sheets,main_sheet_name = remove_main_sheet(all_sheets,ml_summary_table)
    results_dict = {key: None for key in modified_sheets.keys()} # set a result dict with keys as sheet names
    # Retrive from each data frame columns 1,2 (auroc) 4,5(auprc)
    for key,df in modified_sheets.items():
        y_auroc = df.iloc[:,0].values # first column for y values
        x_auroc = df.iloc[:,1].values # second column for x values
        y_auprc = df.iloc[:,3].values
        x_auprc = df.iloc[:,4].values
        wilxco,manwhi = run_statistics(x=x_auroc,y=y_auroc,test_sign=">")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=wilxco,test="wilcoxon_sign",metric="auroc")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=manwhi,test="manwhitenyu",metric="auroc")
        wilxco,manwhi = run_statistics(x=x_auprc,y=y_auprc,test_sign=">")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=wilxco,test="wilcoxon_sign",metric="auprc")
        add_to_results_dict(results_dict=results_dict,name=key,stats_vals=manwhi,test="manwhitenyu",metric="auprc")
    write_dict_to_file(results_dict=results_dict,name=main_sheet_name)
def run_statistics(x,y,test_sign):
    stats1 = get_wilxocon_sign(x=x,y=y,test=test_sign)
    stats2 = get_manwhitenyu(x=x,y=y,test=test_sign)
    return stats1,stats2

def get_manwhitenyu(x,y,test):
    alternative= get_alternative(test)
    x = x.round(5) # 5 decimels after the dot
    y = y.round(5)    
    T_stat,P_val = mannwhitneyu(x=x,y=y,alternative=alternative,method="asymptotic") # asymptotic for more then 8 samples
    stats_dict = {"T_sts":T_stat,"P.v":P_val}
    return stats_dict
def add_to_results_dict(results_dict,name,stats_vals,test,metric):
    metric_dict = {metric : stats_vals} # dict for metric and stats values
    test_dict = {test : metric_dict} 

    if results_dict[name] == None: # key hasnt been setted with a value
        results_dict[name] = test_dict
    else: # update exsiting sub dictionary
        sub_dict = results_dict[name] 
        if test in sub_dict.keys(): # if the test has been made on other data 
            old_metric_dict = sub_dict[test]
            old_metric_dict.update(metric_dict)
        else: sub_dict.update(test_dict) 
        
def write_dict_to_file(results_dict,name):
    file_name = "Statistics_" + name + ".txt"
    with open(file_name, 'w') as file:
        for key, sub_dict in results_dict.items(): # key is the name of data
            for test,metric_dict in sub_dict.items(): # test is the statistic test
                base_str = f"{key}: {test}: "
                added_str = ""
                for metric,stats_val in metric_dict.items(): # metric - type of data
                    added_str = added_str + f"{metric}: {stats_val}, " # stats- are values of the test
                added_str = base_str + added_str + "\n"
                file.write(added_str)



  

         
def remove_main_sheet(sheets_dict,ml_path):
    # Get the main sheet name by the path
    main_sheet_name = ml_path.split("/")[-1].replace(".xlsx","")
    print(f"Dict before: {sheets_dict.keys()}")
    temp_dict = sheets_dict.copy()
    removed_value = temp_dict.pop(main_sheet_name, None)
    if removed_value is not None:
        print(f"Dictionary after removing '{main_sheet_name}': {sheets_dict.keys()}")
    else:
        print(f"Key '{main_sheet_name}' not found in the dictionary.")
    return temp_dict,main_sheet_name
if __name__ == "__main__":
    extract_ml_data("/home/alon/masterfiles/pythonscripts/LOGREG_summary.xlsx")