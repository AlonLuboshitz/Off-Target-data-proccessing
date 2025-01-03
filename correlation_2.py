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
from utilities import  get_feature_name
from Changeseq import get_ending
from file_management import create_folder
from k_groups_utilities import extract_guides_from_partition
# chrom_types are defualt and can be change manauly.
# they correspond to bed files and should be found in the columns of the data
CHROM_TYPES = ['Chromstate']
LABEL = "Label"
READ_COUNT = "GUIDEseq_reads"
CORRELATION_LIST = ["Hypergeo","Spearman","Pearson"]
TARGET = "target"

'''chrom_type - X axis = binary chrom_info e.a - open/close
label/amount - Y axis
function create one data set of:
columns: exp_id_params, corelation_type, p-val,R,R(sqr), phi
runs this corelations on all expriments.'''
def run_stats(path_to_data,list_of_columns,bed_files_path):
    list_of_columns = [("Label","y"),("GUIDEseq_reads","y")]
    data = pd.read_csv(path_to_data)
    # name of dic is type on chrom_info - retrive the name and get the corresponding columns
    chrom_info_columns = get_chrom_info(data,CHROM_TYPES)
    # extend chrom_info_columns to have peak amount and paths
    chrom_info_columns = add_peak_path(chrom_info_columns,bed_files_path)
    # merge the columns list -e.a Y axis, with X axis.
    extened_columns = list_of_columns + chrom_info_columns
    # create x_asis list
    x_axis_list = [name[0] for name in extened_columns if name[1] == "x"]
    # create y_axis_list
    y_axis_list = [name[0] for name in extened_columns if name[1] == "y"]
    # create output folder for correlation
    cor_path = os.path.join(os.path.dirname(path_to_data),f"cor_folder")
    if not os.path.exists(cor_path):
        print(f"Create corelation folder in: {cor_path}")
        os.mkdir(cor_path)
    # create cor_table in cor folder
    
    cor_table, cor_path = create_cor_table(cor_path,name=get_ending(path_to_data))
    # find the params
    # pattern = r'(\d+)params'
    # params = re.search(pattern,path_to_data).group(1)
    # run on each file pearson/spearman/phi
    guides = set(data[TARGET])
    for guide in guides:
        # split the data to only guide data
        guide_data = data[data[TARGET]==guide]
        # get amount of positive and negative
        positive,negative = extract_amount_of_pos_neg(guide_data,label=LABEL)
        cor_table = process_data(data=guide_data,id=guide,params="none",x_axis_list=x_axis_list,y_axis_list=y_axis_list,
                                 list_of_correlations=CORRELATION_LIST,cor_table=cor_table,
                                 positive_amount=positive,negative_amount=negative,columns_info=chrom_info_columns)
        cor_table = add_intersect(cor_table=cor_table,chrom_info=chrom_info_columns,data=guide_data,id=guide)
    # run correlation on whole data
    merged_positive,merged_negative = extract_amount_of_pos_neg(data=data,label=LABEL)
    cor_table = process_data(data=data,id="merged_data",params="None",x_axis_list=x_axis_list,y_axis_list=y_axis_list,
                             list_of_correlations=CORRELATION_LIST,cor_table=cor_table,
                             positive_amount=merged_positive,negative_amount=merged_negative,columns_info=chrom_info_columns)
    # add peak amount
    cor_table = add_peak_amount(chrom_info_columns,cor_table)
    # add intersect amount
    cor_table = add_intersect(cor_table,chrom_info_columns,data=data,id="merged_data")
    # sort by id
    cor_table = cor_table.sort_values(by=["Id","Cor_type","x","y"])
    # save file
    cor_table.to_csv(cor_path,index=False)


'''function to get columns for chrom_info
args - 1. file with data
2. chrom_types
return list of columns with that type and correspoding X axis'''
def get_chrom_info(data,CHROM_TYPES):
    columns = data.columns
    matching_columns = []
    # iterate on chrom types and check for corresponding columns
    for chrom_type in CHROM_TYPES:
        temp_matching_columns = [(col, "x") for col in columns if col.startswith(chrom_type)]
        if not temp_matching_columns:
            print(f"No corresponding columns found for chrom_type: {chrom_type}")
        else:
            matching_columns.extend(temp_matching_columns)
            print(f"Corresponding columns for chrom_type {chrom_type}: {matching_columns}")
    if matching_columns:
        # Filter out elements ending with "_index"
        matching_columns = [(col, val) for col, val in matching_columns if not col.endswith("_index")]    
        return matching_columns

def add_peak_path(columns_tuples,bed_parent_folder):
    # Initialize an empty list to store the processed tuples
    processed_tuples = []
    for column_name, axis in columns_tuples:
        
        # Get the bed path for the current name
        bed_path = get_bed_path_from_name(bed_parent_folder, column_name)
        
        # Read the data frame and get its length
        try:
            df = pd.read_csv(bed_path, sep='\t', header=None)
            peak_amount = len(df)
        except FileNotFoundError:
            peak_amount = None

        # Create a new tuple with the information
        processed_tuple = (column_name, 'x', bed_path, peak_amount)
        
        # Append the processed tuple to the list
        processed_tuples.append(processed_tuple)
    return processed_tuples
        

def get_bed_path_from_name(bed_path,bed_name):
    # split the bed name to parts to create a path.
    # 1 part -> chrom_type
    # 2 + 3 cell type + exp type
    bed_path_parts = bed_name.split("_")
    # concate bed path with chrom type, cell type and exp
    bed_path = os.path.join(bed_path,f'{bed_path_parts[0]}')
    bed_name = f"{bed_path_parts[1]}_{bed_path_parts[2]}"
    # # concate all the rest of the parts to one string and look for the bed file.
    # bed_file_name = ""
    # for i in range(3,len(bed_path_parts) - 1):
    #     bed_file_name = bed_file_name + bed_path_parts[i] + "_"
    # bed_file_name = bed_file_name + bed_path_parts[len(bed_path_parts)-1]
    # look for bed file in bed path
    for dir,subdir,files in os.walk(bed_path):
        for file in files:
            if bed_name in file:
                bed_path = os.path.join(dir,file)
                return bed_path
    


'''this function merge the files given in the path data.'''
def merge_files(file_list,list_of_columns):
    # init data frame for mergning
    merged_df = pd.DataFrame()
    # get columns names
    if list_of_columns:
        selected_columns = [col[0] for col in list_of_columns]
        selected_columns.append("chrinfo_extracted")
        selected_columns.append("Position")
    amount_of_rows = 0
    amount_of_lables = 0
    for file_path in file_list:
        df = pd.read_csv(file_path)
        # sum amount of rows from all data files
        amount_of_rows = amount_of_rows + len(df)
        # add the 1 label amounts to check equality
        amount_of_lables = amount_of_lables + (df['Label_negative'] == 1).sum()
        # create temp df via columns names
        temp_df = df[selected_columns]
        # merge togther with temp
        merged_df = pd.concat([merged_df, temp_df], axis=0, ignore_index=True)
    rows=len(merged_df)
    if not amount_of_rows == rows:
        print("error contanicating rows")
    labels = (merged_df['Label_negative']==1).sum()
    if not amount_of_lables == labels:
        print("error contanicating rows")
    print ("number of total label: {}".format(labels))
    return merged_df

'''function create table for corlation data in spesific path
return tuple of table,path to table.'''
def create_cor_table(path,name):
    file_path = os.path.join(path,f"Cor_data_{name}.csv")
    if os.path.exists(file_path):
        cor_table = pd.read_csv(file_path)
        
    else:
        columns = ['Id','Params','Cor_type','x','y','R','R-Sqr','P-val','Positive_amount','Negative_amount','Peak_amount','intersect_amount']
        cor_table = pd.DataFrame(columns=columns)
        cor_table.to_csv(file_path,index=False)
    
    return (cor_table,file_path)

'''extract amounts of positive off target and negative off targets'''
def extract_amount_of_pos_neg(data,label):
    counts = data[label].value_counts()
    ones_count = counts.get(1, 0)
    zeros_count = counts.get(0, 0)
    return (ones_count, zeros_count)



'''function args:
data - active/inactive - chrom info data
id - name of data
params - type of params with guideseq exp
x,y-axis- axis list to run correlation on
cor_list - names of correlations need to be tested
cor_table - the correlation output to be put in a table.'''
def process_data(data,id, params, x_axis_list, y_axis_list, list_of_correlations, cor_table,positive_amount,negative_amount,columns_info):
    # create empty list for info entered to cor_data_table
    temp_insert_dict = {}
    
    temp_insert_dict["Id"] = id
    temp_insert_dict["Params"] = params
    
    temp_insert_dict['Positive_amount'] = positive_amount
    temp_insert_dict['Negative_amount'] = negative_amount
    
    # make a copy to preserve basic data
    combined_values = temp_insert_dict.copy()
    # iterate axiss and get the series data from each exp
    for cor in list_of_correlations:
        for x in x_axis_list:
    
            x_data = data[x]
            if cor == "Hypergeo":
                added_dict = run_correlation(x_data=x_data,y_data=None, x_name=x, y_name=LABEL, cor_name=cor,data=data,columns_info=columns_info)
                combined_values.update(added_dict)
                temp_values = [combined_values]
                #cor_table = cor_table.append(combined_values,ignore_index=True)
                temp_df = pd.DataFrame(temp_values)
                cor_table = pd.concat([cor_table, temp_df], axis=0, ignore_index=True)
                continue
            for y in y_axis_list:
                
                if y == READ_COUNT:
                    y_data = data[READ_COUNT] = data[READ_COUNT].fillna(0)
                    y_data = y_data.apply(log_transf)
                    y = "Log_" + y
                else: 
                    y_data = data[y]
                # returned values are in a dict {x,y,cor_type,R,r^2,P-val}
                
                added_dict = run_correlation(x_data=x_data,y_data=y_data, x_name=x, y_name=y, cor_name=cor,data=data,columns_info=columns_info)
                combined_values.update(added_dict)
                temp_values = [combined_values]
                temp_df = pd.DataFrame(temp_values)
                cor_table = pd.concat([cor_table, temp_df], axis=0, ignore_index=True)
               

    return cor_table

'''function to run corelation:
args: x,y, data series, x,y names, cor_name
return list of params returned by corelation'''
def run_correlation(x_data,y_data,x_name,y_name,cor_name,data,columns_info):
    returned_params = {}
    returned_params["Cor_type"] = cor_name
    returned_params['x'] = x_name
    returned_params['y'] = y_name
   
    if cor_name == "Pearson":
        correlation, p_value = stats.pearsonr(x_data,y_data)
    elif cor_name == "Spearman":
        correlation, p_value = stats.spearmanr(x_data,y_data)
    elif cor_name == "Hypergeo":
        correlation = 0
        p_value = hypergeometric_test(data,feature=x_name,label_column=y_name)
    
    r_sqr = correlation**2
    returned_params["R"] = correlation
    returned_params["R-Sqr"] =  r_sqr
    returned_params["P-val"] = p_value
    return returned_params

def hypergeometric_test(offtarget_data,feature,label_column,disterbution_stats = None, if_enrichment= False):
    '''This function calculates the hyper geo test for given data set and feature.
    How enrichmet the feature is in the data set. Args:
    1. offtarget data:
    2. feature 
    Disterbution_stats = (active,inactive,sample_size,succesees_in_samples)
    Population - off targets - active + in acttive
    Succses - active of target
    Sample size - chromatin info intergration amount
    succses in sample - active off target via sample size
returns p-val,fold enrichment
 '''
    if disterbution_stats is None:
        # Get active and inactive off targets
        active,inactive = extract_amount_of_pos_neg(offtarget_data,label=label_column)
        # intergration with chrom info
        sample_size = len(offtarget_data[offtarget_data[feature] > 0])
        # succses in sample - where x column(chrom info) is 1 -> open, and y column is 1 (active)
        successes_in_sample = len(offtarget_data[(offtarget_data[feature] >0) & (offtarget_data[label_column] > 0)])
    else :
        active, inactive, sample_size,successes_in_sample = disterbution_stats
    # N population size - off targets
    population_size = active + inactive
    # hypergeo test
    p_value_more_than = 1 - stats.hypergeom.cdf(successes_in_sample - 1, population_size, active, sample_size)
    # enrichments:
    if if_enrichment:
        expected_successes = (active / population_size) * sample_size
        positive_fold_enrichment = successes_in_sample / expected_successes
        # Calculate expected failures
        expected_failures = (inactive/population_size) * sample_size 
        failures_in_sample = sample_size - successes_in_sample
        # Calculate failure fold enrichment
        failure_fold_enrichment = failures_in_sample / expected_failures
        return p_value_more_than,positive_fold_enrichment,failure_fold_enrichment
    return p_value_more_than


''' function transfor information to log of  (1 + val)
+ 1 so log 0 will be define'''
def log_transf(val):
    log_val = np.log(1 + val)
    if log_val == 0 and not val == 0:
        print("error transforming log val, inital was: {}, log: {}".format(val,log_val) )
    return log_val

'''function to get amount of peaks from chromatin information bed files.
args: 1. path to bed files. 
2. list of chromatin info columns using the get_chrom_info
return peak amount for every file'''
def get_peak_amount(path_to_bed,bed_by_column,chrom_type):
   
    # iterate bed_file themself and retrive peak amount for each one.
    line_counts = {}
    
    # Iterate through the directory using os.walk
    for dirpath, _, filenames in os.walk(path_to_bed):
        file_type = ('.bed','.broadPeak','.narrowPeak')
        for filename in filenames:
            if filename.endswith(file_type):
                full_path = os.path.join(dirpath, filename)
                
                # Open the bed file and count the lines
                bed_data = pd.read_csv(full_path, sep='\t', header=None)
                peaks = len(bed_data)
                # add the bed file name the chromtype to match the column
                # Remove '.bed' from the filename
                base_filename = filename[:-4]
                # Create the modified name
                modified_name = chrom_type + "_" + base_filename
                # Store the line count in the dictionary
                line_counts[modified_name] = peaks
    # update bed_by_column tuple list with peaks for each bed file
    for i, (modified_filename, axis) in enumerate(bed_by_column):
        peaks = line_counts.get(modified_filename, None)
        bed_by_column[i] = (modified_filename, axis, peaks)
    return bed_by_column

'''function add the peak amount of each bed file into correspoding column via the x axis.
args: 1. tuple list with name,axis,path,peak
2. the cor_table need to be updated'''
def add_peak_amount(peak_info_by_column,cor_data):
    # Iterate through the tuples and add values to the groups
    for item in peak_info_by_column:
        bed_info, axis, path, peak_value = item
        cor_data.loc[cor_data['x'] == bed_info, 'Peak_amount'] = peak_value

    return cor_data

def add_intersect(cor_table,chrom_info,data,id):
    for item in chrom_info:
        bed_info, axis, path, peak_value = item
        amount = len(data[data[bed_info] >= 1])
        cor_table.loc[(cor_table['x'] == bed_info) & (cor_table["Id"]== id), 'intersect_amount'] = amount
    return cor_table


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
def binary_feature_enrichment_by_partition(partitions, features_columns,label_column,output_path,data_path =None,partition_info_path=None):
    if data_path is None:
        raise RuntimeError("No data path is given")
    if partition_info_path is None:
        raise RuntimeError("No parition data path is given")
    data = pd.read_csv(data_path)
    partition_info = pd.read_csv(partition_info_path)
    output_path = os.path.join(output_path,"Binary")
    create_folder(output_path)
    for partition in partitions:
        partition_guides = extract_guides_from_partition(partition_info,partition)
        partition_data = data[data["target"].isin(partition_guides)]
        temp_path = os.path.join(output_path,f"{partition}_partition.csv")
        binary_feature_enrichment(features_columns,label_column,temp_path,partition_data)
    temp_path = os.path.join(output_path,"All_partitions.csv")
    binary_feature_enrichment(features_columns,label_column,temp_path,data=data)

def binary_feature_enrichment( features_columns, label_column,output_path, data= None, data_path = None):
    '''This function gets a features columns list, label column and data and:
    Creates a data with the enrichment of each feature compared to the label.
    '''
    # Read data
    if data is None:
        if data_path is None:
            raise RuntimeError("No data path is given")
        else:
            data = pd.read_csv(data_path)
            output_path = os.path.join(output_path,f"all_data.csv")
    positives = len(data[data[label_column] > 0])
    negatives = len(data[data[label_column] == 0])
    index = ['positive_peaks','negative_peaks','positive_enrichment','negative_enrichment','p_val','geo_fold_pos','geo_fold_negative','positives','negatives']
    enrichment_data_set = pd.DataFrame()
    enrichment_data_set["Index"] = index
    enrichment_data_set.set_index("Index",inplace=True)
    for feature in features_columns:
        feature_in_positive,feature_in_nagative = get_feature_enrichment(data,feature,label_column)
        total_feature = feature_in_positive + feature_in_nagative
        positive_enrichment = feature_in_positive/positives
        negative_enrichment = feature_in_nagative/negatives
        p_val,geo_fold_positive,geo_fold_negative = hypergeometric_test(None,None,None,(positives,negatives,total_feature,feature_in_positive),True)
        feature_str = get_feature_name(feature)
        enrichment_data_set[feature_str] = [feature_in_positive,feature_in_nagative,positive_enrichment,negative_enrichment,p_val,geo_fold_positive,geo_fold_negative,positives,negatives]
    enrichment_data_set.to_csv(output_path)
def get_feature_enrichment(data,feature, label_column):
    feature_in_positive = len(data[(data[feature] > 0 ) & (data[label_column] > 0)])
    feature_in_nagative = len(data[(data[feature] > 0 ) & (data[label_column] == 0)])
    return (feature_in_positive,feature_in_nagative)

### FEATURE EVALUTION ###
def feature_correlation(data, features_columns, label_column, log_feature = False, log_label = False):
    '''This function will calculate the correlation between the features and the label.
    The function will calculate the pearson correlation and the p value for each feature.
    Args:
    1. data_path - path to the data.
    2. features_columns - columns with the features.
    3. label_column - column with the label.
    4. log_feature - boolean if to log the feature.
    5. log_label - boolean if to log the label.
    ------------
    Returns: dictionary with the features as keys and the (pearson correlation ,p value, x_values, y_values).'''
   
    # Calculate the pearson correlation and p value for each feature
    correlation_dict = {}
    label_values = data[label_column].values
    positive_indices = np.where(label_values > 0)[0]
    positive_label_values = label_values[positive_indices] # only positive values
    if log_label:
        label_values_log = np.log(label_values + 1)
        positive_label_values_log = label_values_log[positive_indices]
    for feature in features_columns:
        # All values
        feature_values = data[feature].values
        r, p = stats.pearsonr(feature_values, label_values)
        correlation_dict[(feature,label_column)] = (r, p, feature_values, label_values)
        # Positive values
        positive_feature_values = feature_values[positive_indices]
        r_pos, p_pos = stats.pearsonr(positive_feature_values, positive_label_values)
        correlation_dict[(feature,f'Positive {label_column}')] = (r_pos, p_pos, positive_feature_values, positive_label_values)
        if log_label:
            # All values log
            r_log, p_log = stats.pearsonr(feature_values, label_values_log)
            correlation_dict[(feature,f'Log {label_column.lower()}')] = (r_log, p_log, feature_values, label_values_log)
            # Positive values log
            r_pos_log, p_pos_log = stats.pearsonr(positive_feature_values, positive_label_values_log)
            correlation_dict[(feature,f'Positive Log {label_column.lower()}')] = (r_pos_log, p_pos_log, positive_feature_values, positive_label_values_log)
    return correlation_dict
def plot_feature_correlation( output_path,  feature_columns, label_column,data_path=None,data=None):
    '''This function will plot the correlation between the features and the label.
    The function will plot the scatter plot for each feature and the label.
    Args:
    1. data_path - path to the data.
    2. output_path - path to save the plots.
    3. feature_columns - columns with the features.
    4. label_column - column with the label.
    ------------
    Returns: None
    
    plot_feature_correlation("/home/dsi/lubosha/Off-Target-data-proccessing/Data/Hendel_lab/merged_gs_caso_onlymism_with_model_scores.csv",
                             "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/Hendel/Feature_correlation/MOFF",["MOFF","GMT"],"Label")'''
    if label_column == "Label":
        log = False
    else: log = True
    # Calculate the correlation
    if data is None:
        if data_path is None:
            raise RuntimeError("No data path is given")
        else:
            data = pd.read_csv(data_path)
    correlation_dict = feature_correlation(data, feature_columns, label_column, log_label = log)
    # Plot the correlation
    for feature_nd_label, values in correlation_dict.items():
        r, p, x_values, y_values = values
        feature,label = feature_nd_label
        feature = feature.replace("_"," ")
        label = label.replace("_"," ")
        y_label = label.replace("Positive","") # remove positive from the label
        plot_correlation(x=x_values, y=y_values, x_axis_label=feature + " score", y_axis_label=y_label, r_coeff=r, p_value=p, title=feature + " " + label, output_path=output_path)


if __name__ == '__main__':
    # run stats create for each guideseq folder the cor_folder
    #run_stats(sys.argv[1],2,sys.argv[2])
    features = [
    'Chromstate_H3K27me3_peaks_binary', 
    'Chromstate_H3K27ac_peaks_binary', 
    'Chromstate_H3K9ac_peaks_binary', 
    'Chromstate_H3K9me3_peaks_binary', 
    'Chromstate_H3K36me3_peaks_binary', 
    'Chromstate_ATAC-seq_peaks_binary', 
    'Chromstate_H3K4me3_peaks_binary', 
    'Chromstate_H3K4me1_peaks_binary'
]
    output_path = "/home/dsi/lubosha/Off-Target-data-proccessing/Feature_correlations/Change-seq/Vivo-vitro"
    data_path ="/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/vivovitro_nobulges_withEpigenetic_indexed_read_count_with_model_scores.csv"
    partition_info = "/home/dsi/lubosha/Off-Target-data-proccessing/Data/Change-seq/partition_guides_78/Changeseq-Partition_vivo_vitro.csv"
                              
    binary_feature_enrichment_by_partition([1,2,3,4,5,6,7],features,"Read_count",output_path,data_path,partition_info)

