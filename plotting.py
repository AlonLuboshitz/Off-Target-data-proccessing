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
        plt.savefig(f'plot_{key}.png')

        # Close the current figure to start a new one for the next key
        plt.close()
'''function to draw some profiles of bw data for positive lables and negative labels'''
def draw_pos_neg_bw_profiles(pos_data_points, neg_data_points, epigenetic_name,window_size):
    # Find the maximum value in all datasets (positive and negative)
    max_value = max(np.max(np.concatenate(pos_data_points)), np.max(np.concatenate(neg_data_points)))
    # Determine the number of sets in positive and negative data
    num_pos_sets = len(pos_data_points)
    num_neg_sets = len(neg_data_points)
    # Create subplots based on the number of sets
    fig, axs = plt.subplots(nrows=num_pos_sets + num_neg_sets, ncols=1, figsize=(8, 8))
    # Plot positive data sets
    # Set x coords by windows size
    x_coords = np.arange(start=0,stop=window_size,step=1)
    for i in range(num_pos_sets):
        axs[i].plot(x_coords, pos_data_points[i], label=f'Positive Set {i + 1}', color='blue')
        axs[i].set_title(f'Positive Set {i + 1} Profile')
        axs[i].set_xlabel('BP')
        axs[i].set_ylabel('Values')
        axs[i].set_ylim([0, max_value])  # Set y-axis limits
        axs[i].legend()
    # Plot negative data sets
    for j in range(num_neg_sets):
        axs[num_pos_sets + j].plot(x_coords,neg_data_points[j], label=f'Negative Set {j + 1}', color='red')
        axs[num_pos_sets + j].set_title(f'Negative Set {j + 1} Profile')
        axs[num_pos_sets + j].set_xlabel('BP')
        axs[num_pos_sets + j].set_ylabel('Values')
        axs[num_pos_sets + j].set_ylim([0, max_value])  # Set y-axis limits
        axs[num_pos_sets + j].legend()
    # Add a common title for the entire figure
    fig.suptitle(f'Epigenetic Profiles - {epigenetic_name}')
    # Add any other details or customization as needed
    # For example, saving the figure or showing it
    plt.savefig(f'{epigenetic_name}_profiles.jpg')
def extract_data_points(data, epigenetic_file, chrom_column, label_column, center_value_column, data_amount,window_size):
    pos_data_sampling = data[data[label_column]==1].sample(data_amount)
    neg_data_sampling = data[data[label_column]==0].sample(data_amount)
    print(f'pos:\n{pos_data_sampling[label_column]}\nneg:\n{neg_data_sampling[label_column]}')
    pos_coords = []
    neg_coords = []
    for center_loc,chrom in zip(pos_data_sampling[center_value_column], pos_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data(epigentic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        pos_coords.append(y_values)
    for center_loc,chrom in zip(neg_data_sampling[center_value_column], neg_data_sampling[chrom_column]): # retive center location, chr
        y_values = get_epi_data(epigentic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        neg_coords.append(y_values)
    return (pos_coords,neg_coords)
def run_pos_neg_profiles(data,file_manager):
    data = pd.read_csv(data)
    for epi_name,epi_file in file_manager.get_bigwig_files():
        pos_coords,neg_coords = extract_data_points(data=data,epigenetic_file=epi_file,chrom_column="chrom",label_column="Label",center_value_column="chromStart",data_amount=5,window_size=2000)
        draw_pos_neg_bw_profiles(pos_coords, neg_coords,epigenetic_name=epi_name,window_size=2000)



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
        y_values = get_epi_data(epigentic_bw_file=epigenetic_file,chrom=chrom,center_loc=center_loc,window_size=window_size)
        sum_values += y_values
        count_values += 1
    average_values = sum_values / count_values
    return average_values
def get_epi_data(epigentic_bw_file, chrom, center_loc, window_size):
    positive_step = negative_step = int(window_size / 2) # set steps to window/2
    if (window_size % 2): # not even
        positive_step += 1 # set pos step +1 (being rounded down before)

    chrom_lim =  epigentic_bw_file.chroms(chrom)
    indices = np.arange(center_loc - negative_step, center_loc + positive_step)
    # Clip the indices to ensure they are within the valid range
    indices = np.clip(indices, 0, chrom_lim - 1)
    # Retrieve the values directly using array slicing
    y_values = epigentic_bw_file.values(chrom, indices[0], indices[-1] + 1)
    
    min_val = epigentic_bw_file.stats(chrom,indices[0],indices[-1] + 1,type="min")[0]  
    # Create pad_values using array slicing
    pad_values_beginning = np.full(max(0, positive_step - center_loc), min_val)
    pad_values_end = np.full(max(0, center_loc + negative_step - chrom_lim), min_val)

    # Combine pad_values with y_values directly using array concatenation
    y_values = np.concatenate([pad_values_beginning, y_values, pad_values_end])
    y_values = y_values.astype(np.float32)
    y_values[np.isnan(y_values)] = min_val # replace nan with min val
    return y_values

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
    file_manager = File_management("pos","neg","bed","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig")
    run_pos_neg_profiles(data="/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_withEpigenetic.csv",file_manager=file_manager)
    #draw_averages_epigenetics()
    #draw_histogram_bigwig(file_manager)
