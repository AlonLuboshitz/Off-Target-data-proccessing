'''This script retrive:
1. Pearson correlation between log (1 + readcounts) to chrom_info
2. Spearnman correlation between log (1 + readcounts) to chrom_info
3. Phi coefficient for binary classification - active(1)/inactive(0) vs chrom_info, e.a open(1)/close(0)
4. add logistic regression
all stats tests run on individual expriments and combined expriments togther.'''
from scipy import stats
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import warnings

'''this function merge the files given in the path data.'''
def merge_files(file_list,list_of_columns):
    
    merged_df = pd.DataFrame()
    if list_of_columns:
        selected_columns = [col[0] for col in list_of_columns]
    amount_of_rows = 0
    amount_of_lables = 0
    for file_path in file_list:
        df = pd.read_csv(file_path)
        # sum amount of rows from all data files
        amount_of_rows = amount_of_rows + len(df)
        # add the 1 label amounts to check equality
        amount_of_lables = amount_of_lables + (df['Label_negative'] == 1).sum()
        temp_df = df[selected_columns]
        merged_df = pd.concat([merged_df, temp_df], axis=0, ignore_index=True)
    rows=len(merged_df)
    if not amount_of_rows == rows:
        print("error contanicating rows")
    # labels = (merged_df['Label_negative']==1).sum()
    # if not amount_of_lables == labels:
    #     print("error contanicating rows")
    # print ("number of total label: {}".format(labels))
    return merged_df

''' function transfor information to log of  (1 + val)
+ 1 so log 0 will be define'''
def log_transf(val):
    log_val = np.log(1 + val)
    if log_val == 0 and not val == 0:
        print("error transforming log val, inital was: {}, log: {}".format(val,log_val) )
    return log_val
'''Function args: 
1. path to data - folder of seperate files
2. list_of_columns - columns need to be kept and find correlation on.
run this steps:
a. on each file in the folder
b. merge all filed togther 
runs 1: pearson and spearman on log + 1 read count transformation vs chrom_info(binary).
2: phi coefficeint on binary inactive/active (0/1) labeling vs chrom_info(binary).
3 creates plots from regression being made with r,r(sqr),p, values.
'''

'''function to get columns for chrom_info
args - 1. file with data
2. chrom_type
return list of columns with that type and correspoding X axis'''
def get_chrom_info(data,chrom_type):
    data = pd.read_csv(data)
    columns = data.columns
    columns_list = [(col,"x") for col in columns if col.startswith(chrom_type)]
    if columns_list:
        return columns_list
    print(f"no columns have been found with {chrom_type} type")
'''chrom_type - X axis = binary chrom_info e.a - open/close
label/amount - Y axis
function create one data set of:
columns: exp_id_params, corelation_type, p-val,R,R(sqr), phi
runs this corelations on all expriments.'''
def run_stats(path_to_data,list_of_columns):
    list_of_columns = [("Label_negative","y"),("bi.sum.mi","y")]
    # get list of files paths:
    files_path = [os.path.join(path_to_data,file) for file in os.listdir(path_to_data)]
    # name of dic is type on chrom_info - retrive the name and get the corresponding columns
    chrom_info_columns = get_chrom_info(files_path[0],os.path.basename(path_to_data))
    # merge the columns list -e.a Y axis, with X axis.
    extened_columns = list_of_columns + chrom_info_columns
    # merge the data via the extened_columns
    merged_data = merge_files(files_path,extened_columns) 
    # create x_asis list
    x_axis_list = [name[0] for name in extened_columns if name[1] == "x"]
    # create y_axis_list
    y_axis_list = [name[0] for name in extened_columns if name[1] == "y"]
    # create output folder for correlation
    cor_path = os.path.join(os.path.dirname(path_to_data),f"cor_folder")
    if not os.path.exists(cor_path):
        print("Create corelation folder")
        os.mkdir(cor_path)
    # create cor_table in cor folder
    cor_table, cor_path = create_cor_table(cor_path)
    # find the params
    pattern = r'(\d+)params'
    params = re.search(pattern,path_to_data).group(1)
    # run on each file pearson/spearman/phi
    list_of_corelations = ["Pearson","Spearman"]
    for path in files_path:
        data = pd.read_csv(path)
        # exp id = targetseq
        id_exp = data.loc[0, 'TargetSequence_negative']
        cor_table = process_data(data,id=id_exp,params=params,x_axis_list=x_axis_list,y_axis_list=y_axis_list,list_of_correlations=list_of_corelations,cor_table=cor_table)
    # run the merged file data
    cor_table = process_data(data=merged_data,id="merged_data",params=params,x_axis_list=x_axis_list,y_axis_list=y_axis_list,list_of_correlations=list_of_corelations,cor_table=cor_table)
    cor_table.to_csv(cor_path,index=False)
    

def process_data(data,id, params, x_axis_list, y_axis_list, list_of_correlations, cor_table):
    # create empty list for info entered to cor_data_table
    temp_insert_dict = {}
    
    temp_insert_dict["Id"] = id
    temp_insert_dict["Params"] = params
    # make a copy to preserve basic data
    combined_values = temp_insert_dict.copy()
    # iterate axiss and get the series data from each exp
    for x in x_axis_list:
        for y in y_axis_list:
            x_data = data[x]
            if y == "bi.sum.mi":
                y_data = data["bi.sum.mi"] = data["bi.sum.mi"].fillna(0)
                y_data = y_data.apply(log_transf)
                y = "Log_" + y
            else: 
                y_data = data[y]
            # returned values are in a dict {x,y,cor_type,R,r^2,P-val}
            for cor in list_of_correlations:
                added_dict = run_correlation(x_data=x_data,y_data=y_data, x_name=x, y_name=y, cor_name=cor)
                combined_values.update(added_dict)
                temp_values = [combined_values]
                #cor_table = cor_table.append(combined_values,ignore_index=True)
                temp_df = pd.DataFrame(temp_values)
                cor_table = pd.concat([cor_table, temp_df], axis=0, ignore_index=True)
    return cor_table


                
'''function to run corelation:
args: x,y, data series, x,y names, cor_name
return list of params returned by corelation'''
def run_correlation(x_data,y_data,x_name,y_name,cor_name):
    returned_params = {}
    returned_params["Cor_type"] = cor_name
    returned_params['x'] = x_name
    returned_params['y'] = y_name
   
    if cor_name == "Pearson":
        correlation, p_value = stats.pearsonr(x_data,y_data)
    elif cor_name == "Spearman":
        correlation, p_value = stats.spearmanr(x_data,y_data)
    r_sqr = correlation**2
    returned_params["R"] = correlation
    returned_params["R-Sqr"] =  r_sqr
    returned_params["P-val"] = p_value
    return returned_params
   
'''function create table for corlation data in spesific path
return tuple of table,path to table.'''
def create_cor_table(path):
    columns = ['Id','Params','Cor_type','x','y','R','R-Sqr','P-val']
    cor_table = pd.DataFrame(columns=columns)
    file_path = os.path.join(path,f"Cor_data.csv")
    cor_table.to_csv(file_path,index=False)
    return (cor_table,file_path)

'''function merge cor_data into one data set from all three guideseq params'''
def merge_cor_data(path):
    df_list = []
    # create list of data frames to merge.
    for folder, subfolder, files in os.walk(path):
        if "cor_folder" in subfolder:
            cor_data_path = os.path.join(folder, "cor_folder", "Cor_data.csv")
            if os.path.isfile(cor_data_path):
                df = pd.read_csv(cor_data_path)  # Read the DataFrame from CSV
                df_list.append(df)  # Add DataFrame to the list
    output_path = os.path.join(path,f"merged_cor_data.csv")
    merged_data = pd.concat(df_list,ignore_index=True)
    merged_data.to_csv(output_path,index=False)
        


''' function creates plot from values, lables, title.
save the figure in the intened path.'''  
def create_scatter_image(x_y_values,title,x_label,y_label,path_for_plot):
    # Create a scatter plot
    plt.scatter(x_y_values[0],x_y_values[1])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title = title + ".png"
    outputpath = os.path.join(path_for_plot,title)
    plt.savefig(outputpath)
''' function gets a path to base folder and creates corelation folder.
inside creates folder for spearman,pearson,phi
returns tuple list of folder (name,path)'''          
def create_cor_dirs(base_path):
    # return one folder back
    base_path = os.path.dirname(base_path)
    # create cor path for cor folder
    cor_path = os.path.join(base_path,f"Cor_folder")
    # check if exsits
    if not os.path.exists(cor_path):
        os.mkdir(cor_path)
    # Create a list of the subdirectory names you want to check for
    subdir_names_to_check = ["Pearson", "Spearman", "Phi"]
    # empty subdirs list - tuples (name,path)
    matching_subdirs = []
    # iterate through list of dirs to check if exists and create them if not
    for subdir_name in subdir_names_to_check:
        subdir_path = os.path.join(cor_path, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        matching_subdirs.append((subdir_name,subdir_path))
    # return the paths for the subdirs.
    return matching_subdirs
    # Y axis = log + 1 transofrmation on  read count.
'''this function merge the files given in the path data.'''

'''Function args:
1. data_path - path to data
2. function_applied - list of tuples - functions,axis to be applied on axiss
3. columns - list of tuples - columns,axis corresponding for each axis
match column to axis and apply function on it.
return axis.'''
def extract_axis(data_path,function_applied,columns):
    # assuming data is csv ',' delimiter.
    data = pd.read_csv(data_path,sep=",",encoding='latin-1',on_bad_lines='skip')
    # Create a dictionary to store the functions by axis name
    functions_by_axis = {axis_name: func for func, axis_name in function_applied}
    # Create a list to store the resulting series
    result_series_list = []
    # Iterate through the columns list
    for column_name, axis_name in columns:
        if axis_name not in functions_by_axis:
            raise ValueError(f"No function specified for axis '{axis_name}'")
        
        custom_function = functions_by_axis[axis_name]
        
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Apply the custom function to the specified column
        result_series = data[column_name].apply(custom_function)
        result_series_list.append((axis_name,result_series))

    return result_series_list

if __name__ == '__main__':
    # run stats create for each guideseq folder the cor_folder
    #run_stats(sys.argv[1],2)
    new_path = os.path.abspath(os.path.join(sys.argv[1], os.pardir, os.pardir))
    merge_cor_data(new_path)
    

