## THIS FILE IS USED TO VALIDATE THE INPUTS FROM THE USER

'''This function is used to validate the input from the user.
 It checks if the input is a valid number and if it is within the range of the list of options
 given by keys of the dictionary'''
def validate_dictionary_input(answer, dictionary):
        assert answer in dictionary.keys(), f"Invalid input. Please choose from {dictionary.keys()}"