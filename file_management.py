import os
import pyBigWig

class File_management:
    # Positive and negative are files paths, pigenetics_bed and bigwig are folders path
    def __init__(self, positives, negatives, epigenetics_bed , bigwig) -> None: 
        self.positive_path = positives
        self.negative_path = negatives
        self.epigenetics_folder_path = epigenetics_bed
        self.bigwig_folder_path = bigwig
        self.create_bigwig_paths()
    ## Getters:
    def get_positive_path(self):
        return self.positive_path
    def get_negative_path(self):
        return self.negative_path
    def get_epigenetics_folder(self):
        return self.epigenetics_folder_path
    def get_bigwig_folder(self):
        return self.bigwig_folder_path
    def get_number_of_bigiwig(self):
        return len(self.bigwig_paths)
    def get_bigwig_paths(self):
        return self.bigwig_paths
    # Functions to create paths from folders
    def create_bigwig_paths(self):
        self.bigwig_paths = []
        for path in os.listdir(self.bigwig_folder_path):
            complete_path = os.path.join(self.bigwig_folder_path,path)
            self.bigwig_paths.append(complete_path)
