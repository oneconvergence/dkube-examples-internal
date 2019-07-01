#!/usr/bin/python3

# Add yello marker to images

import argparse
import os
import sys

import cv2 as cv
import numpy as np


MASK_IMAGE = "mask_image.jpg"
MARKER_BGR_COLOR = [18, 207, 128]


class AddMarker:
    """Add marker to images using Mask image."""

    def __init__(self):
        """Read Mask Image and invert the mask."""
        if not os.path.isfile(MASK_IMAGE):
            print("Error: MASK_IMAGE: %s does not exists." % (MASK_IMAGE))
            sys.exit(0)

        watermark = cv.imread(MASK_IMAGE, cv.IMREAD_UNCHANGED)

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv.cvtColor(watermark, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

        # invert mark -> marker with black color
        self.mask_inv = cv.bitwise_not(mask)

    def add_marker(self, image_name):
        image = cv.imread(image_name)

        # apply black color marker on image
        output = cv.bitwise_or(image, image, mask=self.mask_inv)

        # change marker color to yellow marker color
        output[np.where((output == [0, 0, 0]).all(axis=2))] = \
            MARKER_BGR_COLOR

        cv.imwrite(image_name, output)
        # cv.imshow("image", image)
        # cv.waitKey(0)


def help():
    print("Help: This program add yellow color marker to "
          "images.\n")


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

    obj = AddMarker()
    obj.add_marker(image_name)


if __name__ == '__main__':
    main()
