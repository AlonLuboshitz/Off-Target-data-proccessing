'''
This module is for handeling train/test/validation splits of the features and labels.
'''

from sklearn.model_selection import train_test_split
import numpy as np


def split_to_train_and_val(x, y, task, seed = 42, val_size = 0.1):
    """
    Splits input data into training and validation sets using stratification.

    Args:
        x: Input data (can be a single array or a list of arrays).
        y: Target labels.
        val_size (float, optional): Proportion of data to use for validation (if validation_data is None).

    Returns:
        tuple: (x, y, x_val, y_val) where:
            - x: Training input data.
            - y: Training labels.
            - validation_data: Validation data (generated if not provided).
    """
    if task.lower() == "regression": # create binary bins for the labels
        stratify_mask = (y > 0).astype(int)
    else: stratify_mask = y # Classification already labels
    if isinstance(x,list):
        indices = np.arange(len(y))  # Create indices for the samples
        train_indices, test_indices = train_test_split(indices, test_size=val_size, stratify=stratify_mask, random_state=seed)
        # Split each array in X using the indices
        X_train = [array[train_indices] for array in X_train]
        x_val = [array[test_indices] for array in X_train]
        y_train = y[train_indices]
        y_val = y[test_indices]
    else : # split directly
        X_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size,
                                                                stratify=y, random_state=seed)
    return (X_train,y_train, (x_val, y_val))

def keep_diffrence_guides_indices( guides, test_guides):
        '''This function returns the indexes of the guides that are NOT in the given test_guides 
        Be good to train/test on the guides not presnted in the given guides 
        Args:
        1. guides - list of guides
        2. test_guides - list of guides to keep if exists in guides'''
        return [idx for idx, guide in enumerate(guides) if guide not in test_guides]
def keep_intersect_guides_indices(guides, test_guides):
    '''This function returns the indexes of the guides that are in the given test_guides 
    Be good to train/test on the guides given 
    Args:
    1. guides - list of guides
    2. test_guides - list of guides to keep if exists in guides'''
    return [idx for idx, guide in enumerate(guides) if guide in test_guides]

def split_by_indexes( x_features, y_labels, indices):
    '''
    This function keeps the given indices from the x_features/y_labels lists.
    Args:
    x_features : list of arrays- each array is all (gRNA,OTS) pairs.
    y_label :  list of arrays - each array is the labels for the pairs.
    indices : indices of the arrays to keep.
    -----------
    Returns: x,y concatened arrays.
    '''
    x_, y_ = [], [] 
    
    for idx in indices:
        x_.append(x_features[idx])
        y_.append(y_labels[idx])
    # Concatenate into np array and flatten the y arrays
    x_ = np.concatenate(x_, axis= 0)
    y_ = np.concatenate(y_, axis= 0).ravel()
    return x_, y_


def split_by_guides(guides, guides_t_list, x_features, y_labels):
    '''
    This function splits the data by guides given the guides_t_list.
    For every guide in the guides_t_list, it will keep the data for that guide.
    Args:
    1. guides - list of all the guides in the data
    2. guides_t_list - list of guides to keep
    3. x_features - list of arrays- each array is all (gRNA,OTS) pairs.
    4. y_label -  list of arrays - each array is the labels for the pairs.
    -----------
    Returns: x_train, y_train concataned arrays and the indexes of the guides kept.
    x_train,y_tarin, guides_idx
    '''
    guides_idx = keep_intersect_guides_indices(guides, guides_t_list) # keep only the train/test guides indexes
    if (len(guides_idx) == len(guides)): # All guides are for training
        x_train = np.concatenate(x_features, axis= 0)
        y_train = np.concatenate(y_labels, axis= 0).ravel()
    else:
        x_train, y_train = split_by_indexes(x_features, y_labels, guides_idx) # split by traing indexes
    return x_train, y_train, guides_idx