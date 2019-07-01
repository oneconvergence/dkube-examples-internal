#!/usr/bin/python3

"""
Split Images into 80 20 ratio
- train
      -defective
    - <place 80% of defective tile images here >
      - nondefective
    - <place 80% of nondefective tile images here >
- valid
    -defective
    - <place 15% of defective tile images here >
      - nondefective
    - <place 15% of nondefective tile images here >
- infer 5 %
"""

import argparse
import os
import sys

from preprocessing.add_marker import AddMarker
from preprocessing.remove_marker import RemoveMarker


TRAINED_DIR_NAME = "./data/train"
VALID_DIR_NAME = "./data/valid"
INFER_DIR_NAME = "./infer"
DEFECTIVE_DIR_NAME = "defective"
NONDEFECTIVE_DIR_NAME = "nondefective"
TRAINED_PERCENTAGE = 80
VALID_PERCENTAGE = 15
INFER_PERCENTAGE = 5


class SplitImages:
    def __init__(self):
        # create trained and valid directory
        try:
            os.makedirs(TRAINED_DIR_NAME + "/" + DEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

        try:
            os.makedirs(TRAINED_DIR_NAME + "/" + NONDEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

        try:
            os.makedirs(VALID_DIR_NAME + "/" + DEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

        try:
            os.makedirs(VALID_DIR_NAME + "/" + NONDEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

        try:
            os.makedirs(INFER_DIR_NAME + "/" + DEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

        try:
            os.makedirs(INFER_DIR_NAME + "/" + NONDEFECTIVE_DIR_NAME)
        except FileExistsError:
            pass

    @staticmethod
    def get_percentage(num1, num2):
        return int((num1 * num2) / 100)

    def split_images(self, images_path):
        # print("images_path: ", images_path)
        total_images = len(os.listdir(images_path))

        trained_count = self.get_percentage(TRAINED_PERCENTAGE, total_images)
        valid_count = self.get_percentage(VALID_PERCENTAGE, total_images)
        infer_count = self.get_percentage(INFER_PERCENTAGE, total_images)

        # identify defective and non-defective tile
        if "nondefective" in images_path:
            defective = False
        else:
            defective = True

        # print("defective: ", defective)

        cnt = 1
        for file in os.listdir(images_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                file = "{}/{}".format(images_path, file)
                # copy 80% images to train dir
                if cnt <= trained_count:
                    self.copy_images(file, trained=True, defective=defective)
                elif cnt <= (trained_count + valid_count):
                    # copy 15% images to valid dir
                    self.copy_images(file, trained=False, defective=defective)
                else:
                    # copy 5% images to infer dir
                    self.copy_images(file, infer=True, defective=defective)
            cnt += 1

        if defective:
            print("Successfully Split defective tile images into train "
                  "and eval folder")
        else:
            print("Successfully Split nondefective tile images into train "
                  "and eval folder")
        print("total_count: ", cnt)
        print("trained_count: ", trained_count)
        print("valid_count: ", valid_count)
        print("infer_count: ", infer_count)

    def add_marker(self, images_path):
        """Add marker to good images."""
        obj = AddMarker()

        for file in os.listdir(images_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_name = "{}/{}".format(images_path, file)
                obj.add_marker(image_name)

    def remove_marker(self, images_path):
        """Remove marker from Image using in-painting"""
        obj = RemoveMarker()
        for file in os.listdir(images_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_name = "{}/{}".format(images_path, file)
                obj.remove_marker(image_name)

    @staticmethod
    def copy_images(image_name, trained=True, defective=True, infer=False):
        # print("image_name %s trained: %s defective: %s" % (
        #     image_name, trained, defective))
        cmd = "cp -rf {}".format(image_name)
        if infer:
            cmd = cmd + " {}".format(INFER_DIR_NAME)
        elif trained:
            cmd = cmd + " {}".format(TRAINED_DIR_NAME)
        else:
            cmd = cmd + " {}".format(VALID_DIR_NAME)

        if defective:
            cmd = cmd + "/{}/".format(DEFECTIVE_DIR_NAME)
        else:
            cmd = cmd + "/{}/".format(NONDEFECTIVE_DIR_NAME)
        os.system(cmd)


def help():
    print("Help: This program split images into train and eval directory. "
          "Based on the ratio given: 80/20\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path',
                        help="images directory path")
    parser.add_argument('--add-marker', action="store_true",
                        help="add marker to images")
    parser.add_argument('--remove-marker', action="store_true",
                        help="remove marker from images")
    args = parser.parse_args()
    images_path = args.images_path
    add_marker = args.add_marker
    remove_marker = args.remove_marker

    if not images_path:
        help()
        print("usage: %s [-h] [--images-path IMAGES] "
              "[--add-marker] [--remove-marker]" % (sys.argv[0]))
        return

    obj = SplitImages()
    if add_marker:
        obj.add_marker(images_path)
    elif remove_marker:
        obj.remove_marker(images_path)
    else:
        obj.split_images(images_path)


if __name__ == '__main__':
    main()
