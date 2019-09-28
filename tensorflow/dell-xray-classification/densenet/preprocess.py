#!/usr/bin/python3

"""
Data Preparation from NIH Chest X-ray Dataset of 14 Common Thorax
Disease Categories.
This program create train and valid csv files with link to images.

./data
--> dataset_train.lst
--> dataset_validation.lst
--> ./images
----> image-1.jpg
----> image-2.jpg
"""

import argparse
import os
import sys
import shutil
import tarfile
import time
import zipfile

from itertools import chain
from glob import glob

import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split

DATA_FOLDER = './data'
IMAGES_FOLDER = DATA_FOLDER + "/images"
# INDICES_FILE = './Data_Entry_2017.csv'
# MIN_CASES = 1000
# # zip file of preprocessed data
# INPUT_DATA_DIR = "./data"
OUTPUT_ZIP_FILENAME = "data.zip"
# TRAINING_CSV_FILE = "dataset_train.csv"
# VALID_CSV_FILE = "dataset_validation.csv"


class DataPreparation:
    """DataPreparation Class to prepare Dataset for training."""

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # store images: path key-value pair
        #self.metadata = None

    def run(self):
        """Run Data Preparation steps."""
        # print("input_dir: {}\nOutput_dir: {}".format(
        #     self.input_dir, self.output_dir))

        # 1. Untar all tar files and save all images to DATA_FOLDER
        self.extract_tarfiles()
        # 2. Load the metadata from indices CSV file
        # metadata = self.load_metadata()
        # # 3. Preprocessing of the metadata DataFrames
        # metadata, labels = self.preprocess_metadata(metadata)
        # # 4. Creates a train/test stratification of the dataset
        # train, valid = self.stratify_train_test_split(metadata)

        # # 5. Perform Data Augmentation
        # # # self.data_augmentation()

        # # 6. create dataset and copy all images into images dir
        # self.create_dataset(train, valid)
        # 7. Creating dataset zipfile
        self.zip()
        # 8. Remove dataset
        self.remove_directory()

    def extract_tarfiles(self):
        """
        Untar all TarFiles in the directory and save images to DATA_FOLDER.
        """
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)
        print("\n\n")
        for file in os.listdir(self.input_dir):
            file = os.path.join(self.input_dir, file)
            try:
                if tarfile.is_tarfile(file):
                    print("Extracting TarFile: {} to {}".format(
                        file, DATA_FOLDER))
                    with tarfile.open(file) as opener:
                        opener.extractall(path=IMAGES_FOLDER)
            except IsADirectoryError:
                pass


    @staticmethod
    def remove_directory():
        if os.path.exists(DATA_FOLDER):
            shutil.rmtree(DATA_FOLDER)

    def zip(self):
        """Prepare Dataset Zip file."""
        output_zip_filename = os.path.join(
            self.output_dir, OUTPUT_ZIP_FILENAME)
        print("Creating dataset zip file. ")
        with zipfile.ZipFile(output_zip_filename, 'w') as zip_ref:
            # writing each file one by one
            for root, dirs, files in os.walk(DATA_FOLDER):
                for file in files:
                    zip_ref.write(os.path.join(root, file))
        print("Dataset is available here: %s" % (output_zip_filename))



# time decorator
def timeit(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time()
        print("Total time taken: {:.3f} sec".format(t2 - t1))
    return wrapper


@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        default='{}/{}/'.format(
                            os.getenv("DATUMS_PATH"),
                            os.getenv("DATASET_NAME")),
                        help="video directory path")
    parser.add_argument(
        '--output-dir',
        default='{}/'.format(os.getenv("OUT_DIR", ".")),
        help='Where to save the Output dataset zip file.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir:
        help()
        print("usage: %s [-h] [--input-dir DATASET_DIR]" % (
            sys.argv[0]))
        return

    if not os.path.exists(input_dir):
        print("Error: Please give input path to dataset directory.")
        return

    # Run Preprocessing steps
    dp_obj = DataPreparation(input_dir, output_dir)
    dp_obj.run()


if __name__ == '__main__':
    main()
