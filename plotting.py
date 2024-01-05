import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from file_management import File_management


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
    
    guide_change_dict = get_epigentics_around_center(data,on_column="Label",label_value_list=label_list,center_value_column="chromStart",chrom_column="chrom",file_manager=file_manager)
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
        plt.savefig(f'plot_{key}.png')

        # Close the current figure to start a new one for the next key
        plt.close()

def get_epigentics_around_center(merged_data,on_column,label_value_list,center_value_column,chrom_column,file_manager):
    epigenetics_object = file_manager.get_bigwig_files()
    epi_dict = {}
    for epigeneitc_mark, epigenetic_file in epigenetics_object: # for each epi mark create a list with tuples - (name, average values)
        epi_dict[epigeneitc_mark] = []
        for name,label_value in label_value_list: # for each data points get averages value
            averages = average_epi_around_center(merged_data=merged_data,on_column=on_column,label_value=label_value,center_value_column=center_value_column,chrom_column=chrom_column,epigenetic_file=epigenetic_file)
            epi_dict[epigeneitc_mark].append((name,averages))
    return epi_dict
'''draw +- 10kb range of center off target averages with epigenetics markers'''
def average_epi_around_center(merged_data,on_column,label_value,center_value_column,chrom_column,epigenetic_file):
    # get data points (ots)
    data_points = merged_data[merged_data[on_column]==label_value] # filter data by label
    # Initialize variables for accumulating sum and count
    sum_values = np.zeros(20000)
    count_values = np.zeros(20000)
    for center_loc,chrom in zip(data_points[center_value_column], data_points[chrom_column]): # retive center location, chr
        chrom_lim =  epigenetic_file.chroms(chrom)
        indices = np.arange(center_loc - 10000, center_loc + 10000)
        # Clip the indices to ensure they are within the valid range
        indices = np.clip(indices, 0, chrom_lim - 1)
        # Retrieve the values directly using array slicing
        y_values = epigenetic_file.values(chrom, indices[0], indices[-1] + 1)
       
        min_val = epigenetic_file.stats(chrom,indices[0],indices[-1] + 1,type="min")[0]  
        # Create pad_values using array slicing
        pad_values_beginning = np.full(max(0, 10000 - center_loc), min_val)
        pad_values_end = np.full(max(0, center_loc + 10000 - chrom_lim), min_val)

        # Combine pad_values with y_values directly using array concatenation
        y_values = np.concatenate([pad_values_beginning, y_values, pad_values_end])

        y_values[np.isnan(y_values)] = min_val # replace nan with min val
        # Accumulate sum and count
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



if __name__ == "__main__":
    file_manager = File_management("pos","neg","bed","/home/dsi/lubosha/Off-Target-data-proccessing/Epigenetics/bigwig")

    #draw_averages_epigenetics()
    draw_histogram_bigwig(file_manager)
