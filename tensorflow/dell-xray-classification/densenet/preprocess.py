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

from sklearn.model_selection import train_test_split

DATA_FOLDER = '/tmp/data/images'
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

        os.makedirs(DATA_FOLDER)
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
        #self.remove_directory()

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
                        file, DATA_FOLDER))
                    with tarfile.open(file) as opener:
                        opener.extractall(path=DATA_FOLDER)
            except IsADirectoryError:
                pass

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
    def preprocess_metadata(metadata, minimum_cases=MIN_CASES):
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
        # metadata['Finding Labels'] = metadata['Finding Labels'].map(
        #     lambda x: x.replace('No Finding', ''))

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

        # Drop rows with Column 'Finding Label' have empty value
        metadata = metadata.loc[metadata['Finding Labels'] != '']
        print("Total x-ray records: {}.".format((metadata.shape[0])))

        return metadata, labels

    @staticmethod
    def stratify_train_test_split(metadata):
        """
        Creates a 80/20 train/test stratification of the dataset

        Args:
          metadata: The metadata DataFrame

        Returns:
          train, valid: The stratified train/test DataFrames
        """
        stratify = metadata['Finding Labels'].map(lambda x: x[:4])
        train, valid = train_test_split(metadata,
                                        test_size=0.2,
                                        random_state=2018,
                                        stratify=stratify)
        print("Number of training images: ", train.shape[0])
        print("Number of validation images: ", valid.shape[0])
        return train, valid

    @staticmethod
    def remove_directory():
        if os.path.exists(INPUT_DATA_DIR):
            shutil.rmtree(INPUT_DATA_DIR)

    def prepare_dataset_csv(self, train, valid):
        # dump training dataframe to training file
        # list of columns to write into training file
        column_names = ['Image Index', 'Atelectasis', 'Cardiomegaly',
                        'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                        'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                        'Nodule', 'Pleural_Thickening', 'Pneumonia',
                        'Pneumothorax']

        # Training DataFrame
        dataframe = pd.DataFrame(train)
        dataframe.to_csv(os.path.join(
            INPUT_DATA_DIR, TRAINING_CSV_FILE),
            columns=column_names,
            index=False, float_format=None)
        # Validation DataFrame
        dataframe = pd.DataFrame(valid)
        dataframe.to_csv(os.path.join(
            INPUT_DATA_DIR, VALID_CSV_FILE),
            columns=column_names,
            index=False, float_format=None)

    def create_dataset(self, train, valid):
        """
        Create dataset from train and valid dataframes.
        Copy images to their respective classes.
        """
        print("Preparing train and valid Dataset, this may take time "
              "based on the data available.")

        # create images dir
        if not os.path.exists(os.path.join(INPUT_DATA_DIR, 'images')):
            os.makedirs(os.path.join(INPUT_DATA_DIR, 'images'))

        train = self.copy_images(train)
        valid = self.copy_images(valid)
        # prepare train.csv and valid.csv
        self.prepare_dataset_csv(train, valid)

    def copy_images(self, dataframe):
        """Copy images to images directory."""

        # get dataframe and iterate over dataframes
        dataframe = pd.DataFrame(dataframe)
        dst_dir = os.path.join(INPUT_DATA_DIR, 'images')

        for index_label, image_data in dataframe.iterrows():
            image_index = image_data.get('Image Index')
            image_path = self.metadata.get(image_index)

            # copy image to dst dir
            if image_path:
                shutil.copy(image_path, dst_dir)
                # # update dataframe with relative path
                #new_path = os.path.join(
                #    '.', "/".join(image_path.split('/')[-2:]))
                #dataframe.at[index_label, 'path'] = new_path
        return dataframe

    def zip(self):
        """Prepare Dataset Zip file."""
        output_zip_filename = os.path.join(
            self.output_dir, OUTPUT_ZIP_FILENAME)
        print("Creating dataset zip file. ")
        with zipfile.ZipFile(output_zip_filename, 'w') as zip_ref:
            # writing each file one by one
            for root, dirs, files in os.walk("/tmp/data"):
                for file in files:
                    zip_ref.write(os.path.join(root, file))
        print("Dataset is available here: %s" % (output_zip_filename))

    # @staticmethod
    # def data_augmentation():
    #     """
    #     Data augmentation methods like Resize, Random H-flip and
    #     Random brightness.
    #     """
    #     augment_obj = DataAugmentation()
    #     print("\nPerforming augmentation on all the images.")
    #     augment_obj.perform_augmentation()


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
