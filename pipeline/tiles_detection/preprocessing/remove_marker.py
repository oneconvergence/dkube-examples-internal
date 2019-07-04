#!/usr/bin/python3

# Remove yellow color marker from images using InPainting

import argparse
import sys

import cv2 as cv
import numpy as np


# color in HSV format
YELLOW_HSU_LOWER = [20, 100, 100]
YELLOW_HSU_UPPER = [70, 255, 255]

MARKER_ITERATIONS = 15


class RemoveMarker:
    """Remove marker from bad images using in-painting."""

    def remove_marker(self, marked_image):
        """Remove marker from Image using in-painting"""
        for i in range(MARKER_ITERATIONS):
            self.remove_marker_iterative(marked_image)

    def remove_marker_iterative(self, marked_image):
        """
        Try to remove marker with multiple iterations.
        and saved to new image file.
        """
        image = cv.imread(marked_image)

        # create NumPy arrays from the boundaries
        lower = np.array(YELLOW_HSU_LOWER, dtype="uint8")
        upper = np.array(YELLOW_HSU_UPPER, dtype="uint8")

        # find the colors within the specified boundaries and create
        # the mask
        mask = cv.inRange(image, lower, upper)

        # cv.inpaint(src, inpaintMask, inpaintRadius, flags,)
        output = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)
        cv.imwrite(marked_image, output)

        # # show the images
        # cv.imshow("Original | Modified", np.hstack([image, output]))
        # cv.imshow("mask", mask)
        # cv.waitKey(0)


def help():
    print("Help: This program removes yellow color marker from "
          "images using InPainting.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-name',
                        help="Path to image")
    args = parser.parse_args()
    image_name = args.image_name
    if not image_name:
        help()
        print("usage: %s [-h] [--image-name IMAGE_NAME]" % (sys.argv[0]))
        return

    obj = RemoveMarker()
    obj.remove_marker(image_name)


if __name__ == '__main__':
    main()
