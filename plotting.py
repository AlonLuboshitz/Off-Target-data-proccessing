import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from file_utilities import create_paths
#from file_management import File_management
#from features_engineering import get_epi_data_bw,get_epi_data_bed

def plot_n_rank(n_rank_values, n_tpr_arrays, titles, output_path, general_title):
    if len(n_rank_values) != len(n_tpr_arrays) != len(titles):
        raise ValueError('All input lists must have the same length.')
    #NOTE: check why tpr != 1
    n_tpr_arrays,n_rank_values,titles = argsort_by(n_rank_values, n_tpr_arrays,n_rank_values,titles,descending=True) 

    plt.figure(figsize=(8, 6))
    for i in range(len(n_rank_values)):
        x_values = np.arange(1, len(n_tpr_arrays[i]) + 1)
        plt.plot(x_values, n_tpr_arrays[i], lw=2, label=f'{titles[i]} (N-rank = {n_rank_values[i]:.2f})')
    plt.xlabel('Number of experiments', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve by N experiments')
    plt.legend(loc='lower right',fontsize=11)
    plt.grid(True)
    if not "N_rank" in general_title:
        general_title = general_title + "_N_rank"
    plt.tight_layout()  # Adjust layout to minimize whitespace
    plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
def plot_last_tp(last_tp_index, last_tp_ratio, tpr_arrays, titles, output_path, general_title, positives, negatives):
    '''
    This functions plost the last true positive index and TPR for that point for each model.
    Args:
    1. last_tp_values: (list) of last true positive index values for each model.
    2. tpr_arrays: (list) of true positive rates for each model.
    3. titles: (list) of titles for each model.
    4. output_path: (str) output path for saving the plot.
    5. general_title: (str) general title for the plot.
    6. information_dict: (dict) of information to add to the plot.
    '''
    if len(last_tp_index) != len(last_tp_ratio) != len(tpr_arrays) != len(titles):
        raise ValueError('All input lists must have the same length.')
    # argsort in asecnding order by the last tp values
    last_tp_indices_sorted,tpr_arrays_sorted,titles_sorted = argsort_by(last_tp_index,last_tp_index, tpr_arrays,titles )

    
    # Create a figure
    plt.figure(figsize=(8, 6))
    for i in range(len(last_tp_indices_sorted)):
        x_values = np.arange(1, len(tpr_arrays_sorted[i]) + 1)
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        plt.plot(x_values, tpr_arrays_sorted[i], lw=2,color=color, label=f'{titles_sorted[i]} (Last TP = {last_tp_indices_sorted[i]})')
        plt.axvline(x=last_tp_indices_sorted[i], color=color,lw=1, linestyle='--')

    plt.xlabel('Number of experiments', fontsize=14)
    plt.ylabel('True positive rate', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Last true positive index')
    info_label = f'Positives: {positives}\nTotal: {positives + negatives}'
    plt.plot([], [], ' ', label=info_label)  # Invisible line with empty style

    plt.legend(loc='lower right',fontsize=11)
    plt.grid(True)
    if not "Last_TP" in general_title:
        general_title = general_title + "_Last_TP"
    plt.tight_layout()  # Adjust layout to minimize whitespace
    plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
def plot_roc(fpr_list,tpr_list, aurocs,titles,output_path,general_title):
    '''This function plots the ROC curve for 1 or more models.
    Args:
    1. fpr_list: A list of false positive rates for each model.
    2. tpr_list: A list of true positive rates for each model.
    3. aurocs: A list of AUROC values for each model.
    4. titles: A list of titles for each model.
    5. output_path: A string representing the output path for saving the plot.
    6. general_title: A string representing the general title for the plot.
    ----------
    Show the figure and saves it.'''
    if len(fpr_list) != len(tpr_list) != len(aurocs) != len(titles):
        raise ValueError('All input lists must have the same length.')
    fpr_list,tpr_list,titles,aurocs = argsort_by(aurocs,  fpr_list, tpr_list, titles,aurocs, descending=True)

    plt.figure(figsize=(8, 6))
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2,label=f'{titles[i]} (AUC = {aurocs[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random guess')
    plt.xlabel('False positive rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('True positive rate', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right',fontsize=11)
    plt.grid(True)
    
    if not "AUROC" in general_title:
        general_title = general_title + "_AUROC"
    plt.tight_layout()  # Adjust layout to minimize whitespace
    plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
   
def plot_pr(recall_list, precision_list, auprcs, titles, output_path, general_title):
    '''This function plots the Precision-Recall curve for 1 or more models.
    Args:
    1. recall_list: A list of recall values for each model.
    2. precision_list: A list of precision values for each model.
    3. auprcs: A list of AUPRC values for each model with base line value.
    4. titles: A list of titles for each model.
    5. output_path: A string representing the output path for saving the plot.
    6. general_title: A string representing the general title for the plot.
    ----------
    Show the figure and saves it.'''
    if len(recall_list) != len(precision_list) != len(auprcs) != len(titles):
        raise ValueError('All input lists must have the same length.')
    auprcs_ = [auprc[0] for auprc in auprcs]
    recall_list, precision_list, auprcs, titles = argsort_by(auprcs_,  recall_list, precision_list, auprcs,titles,descending=True)
    plt.figure(figsize=(8, 6))
    for i in range(len(recall_list)):
        plt.plot(recall_list[i], precision_list[i], lw=2,label=f'{titles[i]} (AUC = {auprcs[i][0]:.2f})')
    plt.plot([], [], ' ', label=f'Baseline = {auprcs[0][1]:.5f}')  # Empty plot for baseline legend entry
    plt.xlabel('Recall', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Precision', fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Precision-Recall Curve')
    
    plt.legend(loc='upper right',fontsize=11)
    plt.grid(True)
    if not "AUPRC" in general_title:
        general_title = general_title + "_AUPRC"
    plt.tight_layout()  # Adjust layout to minimize whitespace
    plt.savefig(output_path + f"/{general_title}.png", dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
def plot_correlation(x, y, x_axis_label, y_axis_label, r_coeff, p_value, title, output_path):
    '''This function plots a scatter plot with a linear regression line, and adds the correlation coefficient and p-value to the plot.
    Args:
    1. x: A numpy array representing the x values.
    2. y: A numpy array representing the y values.
    3. x_axis_label: A string representing the x-axis label.
    4. y_axis_label: A string representing the y-axis label.
    5. r_coeff: A float representing the correlation coefficient.
    6. p_value: A float representing the p-value.
    7. title: A string representing the title of the plot.
    8. output_path: A string representing the output path for saving the plot.
    
    ----------
    Show the figure and saves it.'''
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color='blue')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.title(title)
    plt.grid(True)
    plt.xlabel(x_axis_label,fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel(y_axis_label,fontsize=12)
    plt.yticks(fontsize=12)
    num_of_points = len(x)
    plt.text(0.5, 0.9, f'Correlation coefficient: {r_coeff:.2f}\nP-value: {p_value:.2e}\nn = {num_of_points}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
    plt.savefig(output_path + f"/{title}.png")



def argsort_by(argsort_by,  *lists, descending=False):
    argsort_by = np.array(argsort_by)
    indices = np.argsort(argsort_by)
    if descending:
        indices = indices[::-1]
    sorted_lists = []
    for lst in lists:
        sort_lst_ = [lst[i] for i in indices]
        sorted_lists.append(sort_lst_)
    sorted_lists = tuple(sorted_lists)  # Collect sorted lists into a tuple



    return sorted_lists

def plot_binary_feature_heatmap(data_paths, plots_paths):
   
    # Get all data tables paths
    all_tables = create_paths(data_paths)
    all_tables = [(pd.read_csv(table), table.split(".csv")[0].split("/")[-1]) for table in all_tables] 
    all_tables.sort(key=lambda x: x[1])    
    # Number of tables
    num_tables = len(all_tables)

    # Determine grid dimensions for 2 rows
    num_rows = 2
    num_cols = int(np.ceil(num_tables / num_rows))

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 8 * num_rows), sharey=True)
    axes = axes.flatten()  # Flatten for easy indexing

    # Iterate over tables and axes
    for idx, (table_tuple, ax) in enumerate(zip(all_tables, axes)):
        # Extract `geo_fold_pos` and `geo_fold_negative`
        table, table_name = table_tuple
        table.set_index("Index", inplace=True)
        heatmap_data = table.loc[["geo_fold_pos", "geo_fold_negative"]]

        # Transpose the data for heatmap
        heatmap_data = heatmap_data.T

        # Create heatmap rotated for readability
        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            xticklabels=heatmap_data.columns,
            yticklabels=heatmap_data.index,
        )
        ax.set_title(f"{table_name}", fontsize=14)
        ax.set_ylabel("Features", fontsize=12)
        ax.tick_params(axis="x", rotation=90)  # Rotate x-axis labels for better readability

    # Turn off any unused axes
    for ax in axes[num_tables:]:
        ax.axis('off')

    # Set common ylabel

    # Save the plot before calling plt.show()
    output_file = plots_paths + "binary_feature_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_file)



def draw_averages_epigenetics():
    data = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/merged_csgs_withEpigenetic.csv")
    file_manager = File_management("pos","neg","bed","/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/bigwig")
    label_list = [("GUIDE-seq",1),("CHANGE-seq",0)]
    
    guide_change_dict = get_epigentics_around_center(data,on_column="Label",label_value_list=label_list,center_value_column="chromStart",chrom_column="chrom",file_manager=file_manager,window_size=20000)
    label_list = [("CASOFINDER",0)]
    data = pd.read_csv("/home/dsi/lubosha/Off-Target-data-proccessing/merged_csgs_casofinder_withEpigenetic.csv")
    casofinder_dict = get_epigentics_around_center(data,on_column="Label",label_value_list=label_list,center_value_column="chromStart",chrom_column="chrom",file_manager=file_manager)
    merged_dict = {key: guide_change_dict[key] + casofinder_dict[key] for key in guide_change_dict.keys() & casofinder_dict.keys()}
    #  set x coords for -10kb, center, +10 kb
    x_positions = np.linspace(-10000, 10000, 20000)
    colors = ['blue', 'green', 'red']

    # Plot each line
    for key, name_y_values_list in merged_dict.items():
        plt.figure()  # Create a new figure for each key

        for i, (name, y_values) in enumerate(name_y_values_list):
            color = colors[i % len(colors)]  # Cycle through the colors
            plt.plot(x_positions, y_values, label=name, color=color)

        # Set x-axis limits
        plt.xlim(-10000, 10000)
        
        # Add vertical line in ther center      
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Center')
        plt.axvline(x=-10000, color='red', linestyle='-', linewidth=1, label='-10kb')
        plt.axvline(x=10000, color='red', linestyle='-', linewidth=1, label='+10kb')

        # Add legend and labels
        plt.legend()
        plt.xlabel('base pairs')
        plt.ylabel('average values')
        plt.title(f'{key}')
        plt.grid(True)

        # Save the plot
        plt.savefig(f'averages_plot_{key}.png')

        # Close the current figure to start a new one for the next key
        plt.close()
'''function to draw some profiles of bw data for positive lables and negative labels'''
def draw_pos_neg_bw_profiles(pos_data_points, neg_data_points, epigenetic_name,window_size):
    # Find the maximum value in all datasets (positive and negative)
    max_value = max(np.max(np.concatenate(pos_data_points)), np.max(np.concatenate(neg_data_points))) + 0.2
    # Determine the number of sets in positive and negative data
    x_coords = np.arange(start=0,stop=window_size,step=1)

    # Create subplots based on the number of sets
    # Determine the total number of data point sets
    total_sets = len(pos_data_points)

    # Create subplots with a layout determined by the total number of data point sets
    fig, axs = plt.subplots(nrows=total_sets, ncols=2, figsize=(12, 4 * 8))

    # Plot positive data sets in the first column
    for i in range(total_sets):
        axs[i, 0].plot(x_coords, pos_data_points[i], label=f'Positive Set {i + 1}', color='blue')
        axs[i, 0].set_title(f'Positive Set {i + 1} Profile')
        axs[i, 0].set_xlabel('BP')
        axs[i, 0].set_ylabel('Values')
        axs[i, 0].legend()
        axs[i, 0].set_ylim([0, max_value])  # Set y-axis limits

    # Plot negative data sets in the second column
    for j in range(total_sets):
        axs[j, 1].plot(x_coords, neg_data_points[j], label=f'Negative Set {j + 1}', color='red')
        axs[j, 1].set_title(f'Negative Set {j + 1} Profile')
        axs[j, 1].set_xlabel('BP')
        axs[j, 1].set_ylabel('Values')
        axs[j, 1].legend()
        axs[j, 1].set_ylim([0, max_value])  # Set y-axis limits

    # Add a common title for the entire figure
    fig.suptitle(f'Epigenetic Profiles - {epigenetic_name}')

# Adjust layout to prevent overlap
    plt.tight_layout()
    # Add any other details or customization as needed
    # For example, saving the figure or showing it
    plt.savefig(f'{epigenetic_name}_{window_size}_profiles.jpg')
def extract_data_points_bw(data, epigenetic_file, chrom_column, label_column, center_value_column, data_amount,window_size):
    pos_data_sampling = data[data[label_column]==1].sample(data_amount)
    neg_data_sampling = data[data[label_column]==0].sample(data_amount)
    print(f'pos:\n{pos_data_sampling[label_column]}\nneg:\n{neg_data_sampling[label_column]}')
    pos_coords = []
    neg_coords = []
    for center_loc,chrom in zip(pos_data_sampling[center_value_column], pos_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bw(epigenetic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        pos_coords.append(y_values)
    for center_loc,chrom in zip(neg_data_sampling[center_value_column], neg_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bw(epigenetic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        neg_coords.append(y_values)
    return (pos_coords,neg_coords)
def extract_data_points_bed(data, epigenetic_file, chrom_column, label_column, center_value_column, data_amount,window_size):
    pos_data_sampling = data[data[label_column]==1].sample(data_amount)
    neg_data_sampling = data[data[label_column]==0].sample(data_amount)
    print(f'pos:\n{pos_data_sampling[label_column]}\nneg:\n{neg_data_sampling[label_column]}')
    pos_coords = []
    neg_coords = []
    for center_loc,chrom in zip(pos_data_sampling[center_value_column], pos_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bed(epigenetic_bed_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        pos_coords.append(y_values)
    for center_loc,chrom in zip(neg_data_sampling[center_value_column], neg_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bed(epigenetic_bed_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        neg_coords.append(y_values)
    return (pos_coords,neg_coords)
def run_pos_neg_profiles(data,file_manager):
    data = pd.read_csv(data)
    window_size = 10000
    #for epi_name,epi_file in file_manager.get_bigwig_files():
       # pos_coords,neg_coords = extract_data_points_bw(data=data,epigenetic_file=epi_file,chrom_column="chrom",label_column="Label",center_value_column="chromStart",data_amount=10,window_size=window_size)
        #draw_pos_neg_bw_profiles(pos_coords, neg_coords,epigenetic_name=epi_name,window_size=window_size)
    for epi_name,epi_file in file_manager.get_bed_files():
        pos_coords,neg_coords = extract_data_points_bed(data=data,epigenetic_file=epi_file,chrom_column="chrom",label_column="Label",center_value_column="chromStart",data_amount=10,window_size=window_size)
        draw_pos_neg_bw_profiles(pos_coords, neg_coords,epigenetic_name=epi_name,window_size=window_size)


def get_epigentics_around_center(merged_data,on_column,label_value_list,center_value_column,chrom_column,file_manager,window_size):
    epigenetics_object = file_manager.get_bigwig_files()
    epi_dict = {}
    for epigeneitc_mark, epigenetic_file in epigenetics_object: # for each epi mark create a list with tuples - (name, average values)
        epi_dict[epigeneitc_mark] = []
        for name,label_value in label_value_list: # for each data points get averages value
            averages = average_epi_around_center(merged_data=merged_data,on_column=on_column,label_value=label_value,center_value_column=center_value_column,chrom_column=chrom_column,epigenetic_file=epigenetic_file,window_size=window_size)
            epi_dict[epigeneitc_mark].append((name,averages))
    return epi_dict
'''draw +- 10kb range of center off target averages with epigenetics markers'''
def average_epi_around_center(merged_data,on_column,label_value,center_value_column,chrom_column,epigenetic_file,window_size):
    # get data points (ots)
    data_points = merged_data[merged_data[on_column]==label_value] # filter data by label
    # Initialize variables for accumulating sum and count
    sum_values = np.zeros(window_size)
    count_values = np.zeros(window_size)
    for center_loc,chrom in zip(data_points[center_value_column], data_points[chrom_column]): # retive center location, chr
        y_values = get_epi_data_bw(epigenetic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        sum_values += y_values
        count_values += 1
    average_values = sum_values / count_values
    return average_values

def draw_histogram_bigwig(file_manager):
    epigenetics_object = file_manager.get_bigwig_files()
    
    fig, axs = plt.subplots(nrows=len(epigenetics_object), ncols=1, figsize=(8, 4 * len(epigenetics_object)))

    for i, (epigenetic_mark, epigenetic_file) in enumerate(epigenetics_object):
        chr_len = epigenetic_file.chroms("chr7")
        big_wig_values = epigenetic_file.values("chr7",0,chr_len)
        big_wig_values = np.array(big_wig_values)
        big_wig_values[np.isnan(big_wig_values)] = 0.0
        counts, bins = np.histogram(big_wig_values)
        # Plot the histogram in the corresponding subplot
        axs[i].stairs(counts, bins)
        # Customize the subplot if needed (e.g., labels, title, etc.)
        axs[i].set_xticks(bins)
        axs[i].set_xticklabels([f'{bin_val:.2f}' for bin_val in bins])

        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'Histogram for {epigenetic_mark}')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the entire figure to a file
    plt.savefig('epigenetics_histograms.png')
'''Draw a bar plot. y- metric\premonace, x - num of models in the ensemble'''
def plot_ensemeble_preformance(y_values, x_values, title, y_label,x_label,stds,output_path,if_scaling = True, if_ticks = False):
    '''This is a scatter plot function that plots '''
    plt.clf()
    # clear underscores from the x_values
    x_positions = np.arange(len(x_values))
    plt.scatter(x_positions, y_values)
    plt.errorbar(x_positions, y_values, yerr=stds, fmt='none', capsize=5, elinewidth=2, markeredgewidth=2, color='blue')
    
    plt.title(title)
    if if_scaling:
        x_values = [int(x/100) for x in x_values]
        x_label = x_label + " (× 10²)"
    if if_ticks:
        plt.xticks(ticks=x_positions,labels=x_values, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    output_path = output_path + f"/{title}.png"
    plt.savefig(output_path)

def plot_ensemble_performance_mean_std(mean_values, std_values, x_values,p_values, title, y_label, path,partition_information= None):
    plt.clf()
    # Sort indices based on mean values
    sorted_indices = np.argsort(mean_values)
    mean_values_sorted = [mean_values[i] for i in sorted_indices]
    x_values_sorted = [x_values[i] for i in sorted_indices]
    std_sorted = [std_values[i] for i in sorted_indices]
    # get amount of models and set widgth of bars
    num_models = len(mean_values_sorted)
    ind = np.arange(num_models)  # the y locations for the groups
    width = 0.8  # the width of the bars
    # Get the longest label to determine the figure size
    x_singel_labels = [x for x in x_values_sorted if "_" not in x]
    longest_label = max(x_singel_labels, key=len) # Get longest label without considering the subset labels
    label_width = len(longest_label) * 0.2  # Adjust the multiplier as needed for proper spacing

    # Set the figure size based on the width required for the longest label
    fig_width = 8 + label_width  # Adjust the initial figure width as needed
    fig_height = 6  # Adjust the initial figure height as needed

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # create plt
    fig.tight_layout(pad=5)
    bars = ax.barh(ind, mean_values_sorted, width, xerr=std_sorted)
    multi = False
    std_gap = max(std_values) if max(std_values) > 0 else 0.02
    min_x = min(mean_values_sorted) - 3 * std_gap if min(mean_values_sorted) > 0 else 0
    max_x = max(mean_values_sorted) + 2 * std_gap
    # Add p-value annotations
    if p_values: # not empty
        for i, bar in enumerate(bars):
            model = x_values_sorted[i]
            if model == "Only-seq":
                plt.text((bar.get_width()+min_x)/2  - 0.001 , bar.get_y() + (width/2), f'{mean_values_sorted[i]:.3f}', va='center', fontsize=10, color='white')
                continue
            else :
                p_val = p_values[model]
                annotation = p_val_annotation(p_val)
            plt.text(bar.get_width() + std_sorted[i] + 0.001 , bar.get_y() + (width/2), annotation, va='center', fontsize=8)
            plt.text((bar.get_width()+min_x)/2  - 0.001 , bar.get_y() + (width/2), f'{mean_values_sorted[i]:.3f}', va='center', fontsize=10,color='white')

            if "epigenetics" in model:
                multi = True
                bar.set_color('red')
            elif "All" in model:
                bar.set_color('green')
          
       

    ax.set_ylabel('Features', fontsize=12)
    ax.set_xlabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(ind)
    
    ### THIS NOT NEEDED??????
    # Initialize variables for subset labels
    subset_count = 1
    y_labels = []
    subset_mapping = {}
    # Iterate through each model in x_values_sorted
    for model in x_values_sorted:
        if "_" in model:
            subset_label = f'Subset {subset_count}'
            y_labels.append(subset_label)
            subset_mapping[subset_label] = model.split("_")
            subset_count += 1
        else:
            y_labels.append(model)
    ###############################################
    ax.set_yticklabels( y_labels,fontsize = 12)
    
    fig.subplots_adjust(left=label_width/fig_width)
    if multi:
        ax.plot([], label='Epigenetic subsets', color='red')
        for subset_label, subset_models in subset_mapping.items():
            ax.plot([], label=f'{subset_label}: {", ".join(subset_models)}', color='none')
    if partition_information:
        for key,info_ in partition_information.items():
            ax.plot([], label=f'{key}: {info_}', color='none')
    ax.legend(loc='lower right',bbox_to_anchor=(1.05, 0.0),fontsize = 'small',borderaxespad=0.05,ncol=1)
    
    ax.set_xlim(min_x, max_x)
    # Remove right and upper spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

    path = path + f"/{title}.png"
    plt.savefig(path)
def add_pval_legend(plt):
    pval_dict = define_pval_dict()
    for key, value in pval_dict.items():
        plt.plot([], label=f'{key}: {value}', color='none')  # Create an empty plot just for the legend entry
    return plt
def define_pval_dict():
    pval_dict = {}
    pval_dict['***'] = '<0.001'
    pval_dict['**'] = '<0.01'
    pval_dict['*'] = '<0.05'
    pval_dict['ns'] = 'ns'
    return pval_dict
def p_val_annotation(p_val):
    '''Function returns annotation for a given p-value.'''
    if p_val < 0.001:
        annotation = "***"
    elif p_val < 0.01:
        annotation = "**"
    elif p_val < 0.05:
        annotation = "*"
    else:
        annotation = ""
    return annotation
if __name__ == "__main__":
#     #file_manager = File_management("pos","neg","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/Chromstate","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig")
#     #run_pos_neg_profiles(data="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_withEpigenetic.csv",file_manager=file_manager)
#     #draw_averages_epigenetics()
#     #draw_histogram_bigwig(file_manager)
#     import numpy as np
    # scores = np.genfromtxt("/localdata/alon/ML_results/Change-seq/vivo-vitro/Change_seq/CNN/Ensemble/Only_sequence/1_partition/1_partition_50/Combi/ensemble_1.csv", delimiter=',')
    # y_auroc = scores[2:,0]
    # y_auprc = scores[2:,1]
    # y_nrank = scores[2:,2]
    # x = np.arange(2,51)
    # output_ath = "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq"
    # plot_ensemeble_preformance(y_auroc,x,"auroc by models in ensembel","Auroc",output_ath)
    # plot_ensemeble_preformance(y_auprc,x,"auprc by models in ensembel","Auprc",output_ath)
    # plot_ensemeble_preformance(y_nrank,x,"nrank by models in ensembel","N-rank",output_ath)
    pass
    # list_arg = [0.3, 0.7, 0.1, 0.5]  # The list by which to sort
    # from sklearn.metrics import roc_curve  
    # fprs = []
    # tprs =[]
    # for i in range(4):
    #     tpr,fpr,_ = roc_curve([0,1,1,0],[0.1,0.2,0.3,0.4])
    #     tprs.append(tpr)
    #     fprs.append(fpr)
    # list1 = ['a', 'b', 'c', 'd']  # Example list 1

    # list2 = [[1,2,3], np.array(2), np.array(3),np.array(4) ]  # Example list 2
    # list3 = [10, 20, 30, 40]  # Example list 3
    # titles = ['Title1', 'Title2', 'Title3', 'Title4']  # Another list to sort by the same indices
    
    # # Call the function to sort based on `argsort_by`
    # list_arg, fprs,tprs, sorted_list3, sorted_titles = argsort_by(list_arg,list_arg, fprs, tprs, list3, titles, descending=True)
