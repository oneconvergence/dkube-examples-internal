#!/usr/bin/python3

"""
Data Preparation from NIH Chest X-ray Dataset of 14
Common Thorax Disease Categories.
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

from sklearn.model_selection import train_test_split

import preprocessing.params as params
# from preprocessing.data_augmentation import DataAugmentation


class DataPreparation:
    """DataPreparation Class to prepare Dataset for training."""

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # store images: path key-value pair
        self.metadata = None
        self.class_list = None
        self.train_dir = os.path.join(params.INPUT_DATA_DIR, "train")
        self.valid_dir = os.path.join(params.INPUT_DATA_DIR, "valid")

    def run(self):
        """Run Data Preparation steps."""
        # print("input_dir: {}\nOutput_dir: {}".format(
        #     self.input_dir, self.output_dir))

        # # 1. Untar all tar files and save all images to DATA_FOLDER
        self.extract_tarfiles()
        # 2. Load the metadata from indices CSV file
        metadata = self.load_metadata()
        # 3. Preprocessing of the metadata DataFrames
        metadata, labels = self.preprocess_metadata(metadata)
        # 4. Creates a train/test stratification of the dataset
        train, valid = self.stratify_train_test_split(metadata)
        # 5. Create folder for each label to classify images
        self.class_list = labels
        self.create_directory()
        # Perform Data Augmentation
        # self.data_augmentation()

        # # 6. create dataset and copy images into classes
        self.create_dataset(train, valid)
        # # 7. Creating dataset zipfile
        self.zip()
        # # 8. Remove dataset
        self.remove_directory()

    def extract_tarfiles(self):
        """
        Untar all TarFiles in the directory and save images to DATA_FOLDER.
        """
        print("\n\n")
        for file in os.listdir(self.input_dir):
            file = os.path.join(self.input_dir, file)
            try:
                if tarfile.is_tarfile(file):
                    print("Extracting TarFile: {} to {}".format(
                        file, params.DATA_FOLDER))
                    with tarfile.open(file) as opener:
                        opener.extractall(path=params.DATA_FOLDER)
            except IsADirectoryError:
                pass

    def load_metadata(self, data_folder=params.DATA_FOLDER,
                      metadata_file=params.INDICES_FILE):
        """
        Loads the metadata from the indices csv file and scans the
        file system to map the png files to the metadata records.

        Args:
          data_folder: The path to the data folder
          metadata_file: The filename of the metadata csv file

        Returns:
          The metadata DataFrame with a file system mapping of the images
        """

        metadata = pd.read_csv(metadata_file)
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

    @staticmethod
    def preprocess_metadata(metadata, minimum_cases=params.MIN_CASES):
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

        # labels = [c_label for c_label in labels if metadata[c_label].
        #           sum() > minimum_cases]

        sample_weights = metadata['Finding Labels'].map(
            lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2

        sample_weights /= sample_weights.sum()
        metadata = metadata.sample(80000, weights=sample_weights)

        labels_count = [(c_label, int(metadata[c_label].sum()))
                        for c_label in labels]

        print('Labels ({}:{})'.format((len(labels)), (labels_count)))
        # print('Total x-ray records:{}.'.format((metadata.shape[0])))

        return metadata, labels

    @staticmethod
    def stratify_train_test_split(metadata):
        """
        Creates a 70/30 train/test stratification of the dataset

        Args:
          metadata: The metadata DataFrame

        Returns:
          train, valid: The stratified train/test DataFrames
        """
        stratify = metadata['Finding Labels'].map(lambda x: x[:4])
        train, valid = train_test_split(metadata,
                                        test_size=0.3,
                                        random_state=2018,
                                        stratify=stratify)
        print("Number of training images: ", train.shape[0])
        print("Number of validation images: ", valid.shape[0])
        return train, valid

    def create_directory(self):
        """Create directory for each classes inside train and valid."""
        for label in self.class_list:
            if not os.path.exists(os.path.join(self.train_dir, label)):
                os.makedirs(os.path.join(self.train_dir, label))
            if not os.path.exists(os.path.join(self.valid_dir, label)):
                os.makedirs(os.path.join(self.valid_dir, label))

    @staticmethod
    def remove_directory():
        if os.path.exists(params.INPUT_DATA_DIR):
            shutil.rmtree(params.INPUT_DATA_DIR)

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

        for _, image_data in dataframe.iterrows():
            image_index = image_data.get('Image Index')
            labels = image_data.get('Finding Labels').split("|")
            image_path = self.metadata.get(image_index)
            # print(image_index, labels, image_path)

            # copy image to its respective classes inside train dir
            for label in labels:
                if image_path and label in self.class_list:
                    shutil.copy(image_path, os.path.join(
                        dst_dir, label))

    def zip(self):
        """Prepare Dataset Zip file."""
        output_zip_filename = os.path.join(
            self.output_dir, params.OUTPUT_ZIP_FILENAME)
        print("Creating dataset zip file. ")
        with zipfile.ZipFile(output_zip_filename, 'w') as zip_ref:
            # writing each file one by one
            for root, dirs, files in os.walk(params.INPUT_DATA_DIR):
                for file in files:
                    zip_ref.write(os.path.join(root, file))
        print("Dataset is available here: %s" % (output_zip_filename))

    # @staticmethod
    # def data_augmentation():
    #     """
    #     Data augmentation methods like Resize,
    #     """
    #     augment_obj = DataAugmentation()
    #     # Resize all the images
    #     print("\nResizing all the images inside Data Folder: {} "
    #           "to size: {}".format(params.DATA_FOLDER, params.RESIZE_IMAGE))
    #     augment_obj.resize_images()
    #     # # HFlip all the images
    #     # print("\nFlip all the images horizontally.")
    #     # augment_obj.horizontal_flip()
    #     # # adjust Brightness of all the images
    #     # print("\nAdjust brightness of all the images.")
    #     # augment_obj.random_brightness()


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
