import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

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
    data = pd.read_csv("/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs.csv")
    file_manager = File_management("pos","neg","bed","/home/alon/masterfiles/pythonscripts/Changeseq/Epigenetics/bigwig")
    label_list = [("GUIDE-seq",1),("CHANGE-seq",0)]
    guide_change_dict = get_epigentics_around_center(data,on_column="Label",label_value_list=label_list,center_value_column="chromStart",chrom_column="chrom",file_manager=file_manager)
    #label_list = [("CASOFINDER",0)]
    #data = pd.read_csv("/home/alon/masterfiles/pythonscripts/Changeseq/merged_csgs_casofinder_withEpigenetic.csv")
    #casofinder_dict = get_epigentics_around_center(data,on_column="Label",label_value_list=label_list,center_value_column="chromStart",chrom_column="chrom",file_manager=file_manager)
    #merged_dict = {key: guide_change_dict[key] + casofinder_dict[key] for key in guide_change_dict.keys() & casofinder_dict.keys()}
    #  set x coords for -10kb, center, +10 kb
    x_positions = np.linspace(-10000, 10000, 20000)
    colors = ['blue', 'green', 'red']

    # Plot each line
    for key, name_y_values_list in guide_change_dict.items():
        plt.figure()  # Create a new figure for each key

        for i, (name, y_values,std) in enumerate(name_y_values_list):
            # color = colors[i % len(colors)]  # Cycle through the colors
            # plt.plot(x_positions, y_values, label=name, color=color)
            color = plt.cm.viridis(i / len(name_y_values_list))  # Using colormap for varying brightness
            plt.plot(x_positions, y_values, label=name, color=color)
    
            bright_color_plus_std = tuple(c + 0.2 for c in to_rgba(color)[:3])  # Adjust the factor (0.2) as needed
            plt.plot(x_positions, y_values + std, linestyle='dotted',label=f'{name} + std({std:.4f})' ,color=bright_color_plus_std)
            plt.plot(x_positions, y_values - std, linestyle='dotted',label=f'{name} - std({std:.4f})', color=bright_color_plus_std)

    
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
            averages,std = average_epi_around_center(merged_data=merged_data,on_column=on_column,label_value=label_value,center_value_column=center_value_column,chrom_column=chrom_column,epigenetic_file=epigenetic_file)
            epi_dict[epigeneitc_mark].append((name,averages,std))
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
    std = average_values.std()
    return average_values,std
if __name__ == "__main__":
    draw_averages_epigenetics()
