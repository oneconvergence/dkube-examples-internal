#!/usr/bin/python3

# Classify images defective and non-defective based on
# yellow marker present in images.

import argparse
import os
import sys

import cv2 as cv
import numpy as np


GOOD_IMAGES_PATH = "./nondefective/"
BAD_IMAGES_PATH = "./defective/"

# YELLOW_LOWER = [80, 100, 100]
# YELLOW_UPPER = [100, 255, 255]

# yellow marker lower and upper range in HSU
YELLOW_LOWER = [20, 100, 100]
YELLOW_UPPER = [30, 255, 255]


class ImageClassification:
    """Classify image based on yellow marker."""

    def __init__(self):
        if not os.path.exists(GOOD_IMAGES_PATH):
            try:
                os.makedirs(GOOD_IMAGES_PATH)
            except FileExistsError:
                pass
        if not os.path.exists(BAD_IMAGES_PATH):
            try:
                os.makedirs(BAD_IMAGES_PATH)
            except FileExistsError:
                pass

    def yellow_marker_detection(self, image_path):
        """
        Check image is having Yellow color mark.
        Return True is yellow color available else False.
        """
        image = cv.imread(image_path)

        # # Convert BGR to HSV
        # hsv = cv.cvtColor(self.image, cv.COLOR_RGB2HSV)

        # create NumPy arrays from the boundaries
        lower = np.array(YELLOW_LOWER, dtype="uint8")
        upper = np.array(YELLOW_UPPER, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv.inRange(image, lower, upper)
        # count non zero matrix in mask
        # if Yello color is not available count is 0

        # for i in mask:
        #     print(i)
        # print(np.count_nonzero(mask))
        status = False
        if np.count_nonzero(mask) != 0:
            status = True

        # output = cv.bitwise_and(image, image, mask=mask)

        # # show the images
        # cv.imshow("Original | Modified", np.hstack([image, output]))
        # cv.waitKey(0)

        return status

    def classify_images(self, images_path):
        """
        copy non marker images to good folder
        copy marker images to bad folder
        status: True => Bad image
        """
        good_images = 0
        bad_images = 0
        total_images = 0

        # print("images_path: ", images_path)
        for file in os.listdir(images_path):
            file = "{}/{}".format(images_path, file)
            if file.endswith(".jpg") or file.endswith(".JPG"):
                total_images += 1
                status = self.yellow_marker_detection(file)

                # yellow marker : bad image
                if status:
                    bad_images += 1
                    cmd = "cp -r {} {}".format(file, BAD_IMAGES_PATH)
                    # print("Yello marker is available in image: ", file)
                # non yellow marker : good image
                else:
                    good_images += 1
                    cmd = "cp -r {} {}".format(file, GOOD_IMAGES_PATH)
                    # print("Yello marker is not available in image: ", file)
                os.system(cmd)

        print("\nNon-Defective images stored under directory: %s\n"
              "Defective images stored under directory: %s.\n" % (
                GOOD_IMAGES_PATH, BAD_IMAGES_PATH))

        print("Total Images: ", total_images)
        print("Non-Defective Images: ", good_images)
        print("Defective Images: ", bad_images)


def help():
    print("Help: This program classify images as defective and non-defective "
          "based on yellow marker present in images.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path',
                        help="images directory path")
    args = parser.parse_args()
    images_path = args.images_path
    if not images_path:
        help()
        print("usage: %s [-h] [--images-path IMAGES_PATH]" % (
            sys.argv[0]))
        return

    obj = ImageClassification()
    obj.classify_images(images_path)


if __name__ == '__main__':
    main()
