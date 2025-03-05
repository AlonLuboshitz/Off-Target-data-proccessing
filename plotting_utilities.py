from matplotlib import pyplot as plt
import numpy as np

def return_colormap(list1, list2=None):
        
    cmap = plt.get_cmap("tab10")

    # Assign colors to the first list
    color_map = {element: cmap(i / len(list1)) for i, element in enumerate(list1)}
    if list2:
        for i, element in enumerate(list2):
            if element not in color_map:
                color_map[element] = cmap(i / len(list2))  # Assign new colors only for new elements
    return color_map
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

def render_sg_ot_to_positions(sgrna,ot):
    '''
    This function renders the sgRNA and OT sequences to positions on the heatmap.
    It will return a list of 24 positions and labels for the heatmap.
    Args:
    1. sgrna: (str) - the sgRNA sequence.
    2. ot: (str) - the OT sequence.
    -----------
    Returns: list of positions and labels for the heatmap.'''
    positions = []
    labels = [""] * 24  # Initialize a list of 24 empty labels
    seq_length = len(sgrna)  # Determine the length of the sequence
    if seq_length not in [23, 24]:
        raise ValueError("Each sequence must be either 23 or 24 characters long")
    start_index = 0 if seq_length == 24 else 1  # Shift by 1 if length is 23
    for i in range(seq_length):
        pos = start_index + i
        positions.append(pos)
        labels[pos] = f"{sgrna[i]}\n{ot[i]}"  # Assign formatted labels
    return list(range(24)), labels  # Return fixed 24 positions
