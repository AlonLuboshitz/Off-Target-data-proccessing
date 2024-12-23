'''
This module contains setting parameters and validating them.
'''

def with_bulges(constraints):
    '''
    Return True if the constraints allows for bulges
    '''
    if constraints == 1: # No constraints
        return True 
    elif constraints == 2: # Mismatch only
        return False
    elif constraints == 3: # Bulges only
        return True
    else:
        raise ValueError("Invalid constraints value")

def get_ot_constraint_name(ot_constraint):
    '''
    Get the name of the off-target constraint.
    '''
    if ot_constraint == 1:
        return "No constraints"
    elif ot_constraint == 2:
        return "Mismatch only"
    elif ot_constraint == 3:
        return "Bulges only"
    else:
        raise ValueError("Invalid ot_constraint value")

