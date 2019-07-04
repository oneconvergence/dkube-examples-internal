#!/usr/bin/python3

"""
Preprocessing:
prepare datasets from videos
1. extract frames from videos and store into INPUT_IMAGES_DIR
2. Classify Images into 2 dir classes defective and nondefective
3. Remove marker from defective images
4. split images into 80 20 ratio
5. Prepare a ZIP file
"""

import argparse
import os
import sys
import shutil
import time
import zipfile

from video_extraction.tiles_videos_extraction import VideoExtraction
from preprocessing.image_classification import ImageClassification
from preprocessing.data_augmentation import DataAugmentation
from preprocessing.split_images import SplitImages

# output directory to store cropped images extracted from videos
OUTPUT_DIR = "./preprocessed_data"
INPUT_IMAGES_DIR = OUTPUT_DIR + "/data"
GOOD_IMAGES_PATH = "./nondefective/"
BAD_IMAGES_PATH = "./defective/"
# input zip dir: where preprocessed data present
INPUT_ZIP_DIR = "./data/"
# zip file of preprocessed data
OUTPUT_ZIP_FILENAME = "data.zip"


class PreProcessing:

    def run(self, video_path, saved_dataset_dir="."):
        """Run preprocessing steps."""

        # 1. extract frames from video files
        self.extract_frames(video_path)
        # 2. classify images
        self.classifyImages()
        # 3. remove marker from defective images
        self.remove_marker()
        # 4. data augmentation of defective images
        self.data_augmentation()
        # 5. split images into 80 20 ratio
        self.split()
        # 6. Prepare a ZIP file
        self.zip(saved_dataset_dir)
        # 7. Remove un-necessary folders
        self.clean()

    def extract_frames(self, video_path):
        print("\n1. Extracting frames from videos")
        # extract frames from videos
        frames_capture = VideoExtraction()
        frames_capture.FrameCapture(video_path)

    def classifyImages(self):
        print("\n2. Classifying images into 2 classes defective and "
              "nondefective")
        classify_obj = ImageClassification()
        classify_obj.classify_images(INPUT_IMAGES_DIR)

    def remove_marker(self):
        print("\n3. Removing Marker from defective images")
        rm_obj = SplitImages()
        rm_obj.remove_marker(BAD_IMAGES_PATH)

    def data_augmentation(self):
        print("\n4. Data augmentation of defective images")
        da_obj = DataAugmentation()
        da_obj.flip_images(BAD_IMAGES_PATH, BAD_IMAGES_PATH)

    def split(self):
        print("\n5. Split Images into train and eval folder")
        obj = SplitImages()
        obj.split_images(BAD_IMAGES_PATH)
        obj.split_images(GOOD_IMAGES_PATH)

    def zip(self, saved_dataset_dir):
        output_zip_filename = ("{}/{}".format(
            saved_dataset_dir, OUTPUT_ZIP_FILENAME))

        print("6. Creating dataset zip file.")
        with zipfile.ZipFile(output_zip_filename, 'w') as zip_ref:
            # writing each file one by one
            for root, dirs, files in os.walk(INPUT_ZIP_DIR):
                for file in files:
                    zip_ref.write(os.path.join(root, file))
        print("Dataset is available here: %s" % (output_zip_filename))

    def clean(self):
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(GOOD_IMAGES_PATH)
        shutil.rmtree(BAD_IMAGES_PATH)
        shutil.rmtree(INPUT_ZIP_DIR)
        # try:
        #     cmd = "rm -r %s/train/" % (INPUT_ZIP_DIR)
        #     os.system(cmd)
        #     cmd = "rm -r %s/valid/" % (INPUT_ZIP_DIR)
        #     os.system(cmd)
        # except Exception:
        #     pass
        try:
            shutil.rmtree("./infer")
        except FileNotFoundError:
            pass


def help():
    print("Help: This program performs PreProcessing steps and prepares "
          "dataset from video files.\n")


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
    parser.add_argument('--video-dir',
                        default='{}/{}/'.format(
                            os.getenv("DATUMS_PATH"),
                            os.getenv("DATASET_NAME")),
                        help="video directory path")
    parser.add_argument(
        '--saved_dataset_dir',
        default='{}/'.format(os.getenv("OUT_DIR", ".")),
        help='Where to save the dataset zip file.')
    args = parser.parse_args()
    video_dir = args.video_dir
    saved_dataset_dir = args.saved_dataset_dir

    if not video_dir:
        help()
        print("usage: %s [-h] [--video-dir VIDEO_DIR]" % (
            sys.argv[0]))
        return

    if not os.path.exists(video_dir):
        print("Please give path to video directory.")
        return

    # Run Preprocessing steps
    pre_obj = PreProcessing()
    pre_obj.run(video_dir, saved_dataset_dir)


if __name__ == '__main__':
    main()
