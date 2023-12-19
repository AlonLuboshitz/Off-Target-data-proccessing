import os
import pyBigWig

class File_management:
    # Positive and negative are files paths, pigenetics_bed and bigwig are folders path
    def __init__(self, positives, negatives, epigenetics_bed , bigwig) -> None: 
        self.positive_path = positives
        self.negative_path = negatives
        self.epigenetics_folder_path = epigenetics_bed
        self.bigwig_folder_path = bigwig
        self.create_bigwig_files_objects()
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
        return len(self.bigwig_files)
   
    def get_bigwig_files(self):
        if len(self.bigwig_files) > 0 :
            return self.bigwig_files
        else: raise ValueError("No bigwig files")
    # Functions to create paths from folders
    '''Create paths list from folder'''
    def create_paths(self,folder):
        paths = []
        for path in os.listdir(folder):
            paths.append(os.path.join(folder,path))
        return paths

    '''Create pyBigWig objects list of the bigwig files'''
    def create_bigwig_files_objects(self):
        self.bigwig_files = []
        for path in self.create_paths(self.bigwig_folder_path):
            name = path.split("/")[-1].split(".")[0] # retain the name of the file (includes the marker)
            name_object_tpl = (name,pyBigWig.open(path))
            self.bigwig_files.append(name_object_tpl)
        return self.bigwig_files
