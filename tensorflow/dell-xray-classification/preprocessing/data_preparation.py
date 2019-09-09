#!/usr/bin/python3

"""
Data Preparation from NIH Chest X-ray Dataset of 14
Common Thorax Disease Categories.
"""

import argparse
# import multiprocessing
import os
import sys
import shutil
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd
from itertools import chain
from glob import glob
from sklearn.model_selection import train_test_split

# CONSTANTS
DATA_FOLDER = '/tmp/data/'
INDICES_FILE = './Data_Entry_2017.csv'
MIN_CASES = 1000
# resizes the image to 224 x 224
RESIZE_IMAGE = (224, 224)
# zip file of preprocessed data
INPUT_DATA_DIR = "./data"
OUTPUT_ZIP_FILENAME = "data.zip"


class DataPreparation:
    """DataPreparation Class to prepare Dataset for training."""

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # store images: path key-value pair
        self.metadata = None
        self.classes = None
        self.train_dir = os.path.join(INPUT_DATA_DIR, "train")
        self.valid_dir = os.path.join(INPUT_DATA_DIR, "valid")

    def run(self):
        """Run Data Preparation steps."""
        # print("input_dir: {}\nOutput_dir: {}".format(
        #     self.input_dir, self.output_dir))

        # 1. Untar all tar files and save all images to DATA_FOLDER
        self.extract_tarfiles()
        # 2. Load the metadata from indices CSV file
        metadata = self.load_metadata()
        # 3. Preprocessing of the metadata DataFrames
        metadata, labels = self.preprocess_metadata(metadata)
        # 4. Creates a train/test stratification of the dataset
        train, valid = self.stratify_train_test_split(metadata)
        # 5. Create folder for each label to classify images
        self.classes = labels
        self.create_directory()
        # 6. create dataset and copy images into classes
        self.create_dataset(train, valid)
        # 7. Creating dataset zipfile
        self.zip()
        # 8. Remove dataset
        self.remove_directory()

    # @staticmethod
    # def untar(tar_file):
    #     print("tarfile: ", tar_file)
    #     with tarfile.open(tar_file) as opener:
    #         opener.extractall(path=DATA_FOLDER)

    def extract_tarfiles(self):
        """
        Untar all TarFiles in the directory and save images to DATA_FOLDER.
        """
        for file in os.listdir(self.input_dir):
            try:
                if tarfile.is_tarfile(file):
                    print("Extracting TarFile: {} to {}".format(
                        file, DATA_FOLDER))
                    with tarfile.open(file) as opener:
                        opener.extractall(path=DATA_FOLDER)
            except IsADirectoryError:
                pass

        # # with multiprocessing
        # with multiprocessing.Pool() as pool:
        #     pool.map(self.untar, os.listdir(self.input_dir))

    def load_metadata(self, data_folder=DATA_FOLDER,
                      metadata_file=INDICES_FILE):
        """
        Loads the metadata from the indices csv file and scans the
        file system to map the png files to the metadata records.

        Args:
          data_folder: The path to the data folder
          metadata_file: The filename of the metadata csv file

        Returns:
          The metadata DataFrame with a file system mapping of the images
        """

        metadata = pd.read_csv(os.path.join(self.input_dir, metadata_file))
        file_system_scan = {os.path.basename(x): x for x in
                            glob(os.path.join(data_folder, 'images', '*.png'))}
        # if len(file_system_scan) != metadata.shape[0]:
        #     raise Exception(
        #         'ERROR: Different number metadata records and png files.')

        metadata['path'] = metadata['Image Index'].map(file_system_scan.get)
        print('Total x-ray records:{}.'.format((metadata.shape[0])))

        # store images path as key-value pair
        self.metadata = file_system_scan

        return metadata

    def preprocess_metadata(self, metadata, minimum_cases=MIN_CASES):
        """
        Preprocessing of the metadata df.
        We remove the 'No Finding' records and all labels with less
        than minimum_cases records.

        Args:
          metadata: The metadata DataFrame

        Returns:
          metadata, labels : The preprocessed metadata DataFrame and the
          valid labels left in the metadata.
        """
        # remove the 'No Finding' records
        metadata['Finding Labels'] = metadata['Finding Labels'].map(
            lambda x: x.replace('No Finding', ''))

        # extract labels from metadata
        labels = np.unique(
            list(chain(*metadata['Finding Labels'].map(
                lambda x: x.split('|')).tolist())))
        labels = [x for x in labels if len(x) > 0]

        for c_label in labels:
            if len(c_label) > 1:  # leave out empty labels
                metadata[c_label] = metadata['Finding Labels'].map(
                    lambda finding: 1.0 if c_label in finding else 0)

        labels = [c_label for c_label in labels if metadata[c_label].
                  sum() > minimum_cases]

        sample_weights = metadata['Finding Labels'].map(
            lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2

        sample_weights /= sample_weights.sum()
        metadata = metadata.sample(80000, weights=sample_weights)

        labels_count = [(c_label, int(metadata[c_label].sum()))
                        for c_label in labels]

        print('Labels ({}:{})'.format((len(labels)), (labels_count)))
        # print('Total x-ray records:{}.'.format((metadata.shape[0])))

        return metadata, labels

    def stratify_train_test_split(self, metadata):
        """
        Creates a train/test stratification of the dataset

        Args:
          metadata: The metadata DataFrame

        Returns:
          train, valid: The stratified train/test DataFrames
        """
        stratify = metadata['Finding Labels'].map(lambda x: x[:4])
        train, valid = train_test_split(metadata,
                                        test_size=0.25,
                                        random_state=2018,
                                        stratify=stratify)
        # print("train: {}\nvalid: {}".format(train, valid))
        return train, valid

    def create_directory(self):
        """Create directory for each classes inside train and valid."""
        for label in self.classes:
            if not os.path.exists(os.path.join(self.train_dir, label)):
                os.makedirs(os.path.join(self.train_dir, label))
            if not os.path.exists(os.path.join(self.valid_dir, label)):
                os.makedirs(os.path.join(self.valid_dir, label))

    def remove_directory(self):
        if os.path.exists(INPUT_DATA_DIR):
            shutil.rmtree(INPUT_DATA_DIR)

    def create_dataset(self, train, valid):
        """
        Create dataset from train and valid dataframes.
        Copy images to their respective classes.
        """
        print("Preparing train and valid Dataset, this may take time "
              "based on the data available.")
        self.copy_images(train=train)
        self.copy_images(valid=valid)

    def copy_images(self, train=None, valid=None):
        """Copy images to their respective classes."""

        # get dataframe and iterate over dataframes
        if train is not None:
            dst_dir = self.train_dir
            dataframe = pd.DataFrame(train)
        elif valid is not None:
            dst_dir = self.valid_dir
            dataframe = pd.DataFrame(valid)

        for index, image_data in dataframe.iterrows():
            image_index = image_data.get('Image Index')
            labels = image_data.get('Finding Labels').split("|")
            image_path = self.metadata.get(image_index)
            # print(image_index, labels, image_path)

            # copy image to its respective classes inside train dir
            for label in labels:
                if image_path and label in self.classes:
                    shutil.copy(image_path, os.path.join(
                        dst_dir, label))

    def zip(self):
        output_zip_filename = os.path.join(
            self.output_dir, OUTPUT_ZIP_FILENAME)
        print("Creating dataset zip file. ")
        with zipfile.ZipFile(output_zip_filename, 'w') as zip_ref:
            # writing each file one by one
            for root, dirs, files in os.walk(INPUT_DATA_DIR):
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
