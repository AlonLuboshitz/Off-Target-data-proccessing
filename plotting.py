import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from file_management import File_management
#from features_engineering import get_epi_data_bw,get_epi_data_bed

'''given a x data and y data draws a auc curve.
add marking lines to x_pinpoint and y_points - i.a y = 0.5 draw a line there to mark this value
y_data is a dictionary - key: (expirement name\guide rna), value: ascending rates
plot all keys through the x data
x data is a dictionary with one key: name of label, value: data '''
def draw_auc_curve(x_data,y_data,x_pinpoints,y_pinpoints,title):
    # Plot the AUC curve
    plt.figure()

    for key_x,ranks in x_data.items():   
        for key_y,rates in y_data.items():
            plt.plot(ranks, rates,label=key_y)
    plt.xlabel(key_x)
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.show()
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
def plot_ensemeble_preformance(y_values, x_values, title, y_label,path):
    plt.clf()
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel('num_models')
    plt.ylabel(y_label)
    path = path + f"/{title}.png"
    plt.savefig(path)

def plot_ensemble_performance_mean_std(mean_values, std_values, x_values,p_values, title, y_label, path):
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
    longest_label = max(x_values_sorted, key=len)
    label_width = len(longest_label) * 0.1  # Adjust the multiplier as needed for proper spacing

    # Set the figure size based on the width required for the longest label
    fig_width = 8 + label_width  # Adjust the initial figure width as needed
    fig_height = 6  # Adjust the initial figure height as needed

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # create plt
    fig.tight_layout(pad=5)
    bars = ax.barh(ind, mean_values_sorted, width, xerr=std_sorted)
    multi = False
    min_x = min(mean_values_sorted) - 3 * (max(std_values)) if min(mean_values_sorted) > 0 else 0
    max_x = max(mean_values_sorted) + 2 * (max(std_values))
    # Add p-value annotations
    if p_values: # not empty
        for i, bar in enumerate(bars):
            model = x_values_sorted[i]
            if model == "Only-seq":
                plt.text((bar.get_width()+min_x)/2  - 0.001 , bar.get_y() + (width/2), f'{mean_values_sorted[i]:.4f}', va='center', fontsize=8, color='white')
                continue
            else :
                p_val = p_values[model]
                annotation = p_val_annotation(p_val)
            plt.text(bar.get_width() + std_sorted[i] + 0.001 , bar.get_y() + (width/2), annotation, va='center', fontsize=8)
            plt.text((bar.get_width()+min_x)/2  - 0.001 , bar.get_y() + (width/2), f'{mean_values_sorted[i]:.4f}', va='center', fontsize=8,color='white')

            if "_" in model or model == "All":
                multi = True
                bar.set_color('red')
          
       

    ax.set_ylabel('Epigenetic marks', fontsize=12)
    ax.set_xlabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(ind)
    # If models in x_values_sorted are with _ in them turn into list of strings
    y_labels = [model.split("_") if "_" in model else model for model in x_values_sorted]
    y_labels_joined = ['\n'.join(label) if isinstance(label, list) else label for label in y_labels]
    ax.set_yticklabels( y_labels_joined,fontsize = 8)  # Use the sorted x_values as labels
    

    
    
    fig.subplots_adjust(left=label_width/fig_width)
    if multi:
        ax.plot([], label='Epigenetic subsets', color='red')
    ax.legend(loc='lower right')
    
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
# if __name__ == "__main__":
#     #file_manager = File_management("pos","neg","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/Chromstate","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig")
#     #run_pos_neg_profiles(data="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_withEpigenetic.csv",file_manager=file_manager)
#     #draw_averages_epigenetics()
#     #draw_histogram_bigwig(file_manager)
#     import numpy as np
#     scores = np.genfromtxt("/home/dsi/lubosha/Off-Target-data-proccessing/ML_results/Change_seq/Ensembles/1_partition_50/Combi/ensemble_1.csv", delimiter=',')
#     y_auroc = scores[2:,0]
#     y_auprc = scores[2:,1]
#     y_nrank = scores[2:,2]
#     x = np.arange(2,51)
#     output_ath = "/home/dsi/lubosha/Off-Target-data-proccessing/Plots/ensembles/change_seq"
#     plot_ensemeble_preformance(y_auroc,x,"auroc by models in ensembel","Auroc",output_ath)
#     plot_ensemeble_preformance(y_auprc,x,"auprc by models in ensembel","Auprc",output_ath)
#     plot_ensemeble_preformance(y_nrank,x,"nrank by models in ensembel","N-rank",output_ath)
