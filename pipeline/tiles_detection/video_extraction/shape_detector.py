#!/usr/bin/python3

"""
Identify Square or rectangular shape in images using Cosine Angle.
Crop the identified tile images and resize it to 512 * 512.
Save the images under CROPPED_IMAGES_PATH directory.
"""


import argparse
import os
import sys

import cv2 as cv
import numpy as np

# output directory to store cropped images
CROPPED_IMAGES_PATH = "./preprocessed_data" + "/data"
RESIZE_WIDTH_HEIGHT = (512, 512)  # square image
# minimum ans maximum area needed to detect tile if Image size is 512 * 512
MIN_AREA_512 = 100000
MAX_AREA_512 = 180000
# minimum and maximum area needed to detect tile if Image size is 1080 * 1080
MIN_AREA_1080 = 590000
MAX_AREA_1080 = 700000
# Maximum cosine angle to identify rectangle or square
MAX_COS = 0.1


class ShapeDetector:
    """
    Detect Shape in images and save to cropped_images directory.
    Input: Image Directory Path
    Output: Save images to cropped images directory
    """
    def __init__(self):
        # create output directory to store cropped images
        if not os.path.exists(CROPPED_IMAGES_PATH):
            try:
                os.makedirs(CROPPED_IMAGES_PATH)
            except FileExistsError:
                pass

    def display_image(self, orig_image, modified_image=None):
        # show the images
        if modified_image is not None:
            cv.imshow("Original | Modified", np.hstack(
                [orig_image, modified_image]))
        else:
            cv.imshow("Original", orig_image)
        cv.waitKey(0)

    def detect(self, images_path):
        """
        Detect square in image and draw it.
        Input: Path to input Images
        Output: Save cropped image
        """
        # to save crop images for each video into separate directory
        # video_dir_name = images_path.split("/")[-1]
        # # save crop images for each video into separate directory
        # cropped_images_path = CROPPED_IMAGES_PATH + "/{}".format(
        #     video_dir_name)
        # if not os.path.exists(cropped_images_path):
        #     try:
        #         os.makedirs(cropped_images_path)
        #     except FileExistsError:
        #         pass

        for file in os.listdir(images_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image = "{}/{}".format(images_path, file)
                self.detect_square_shape(image)

    def save_image(self, image_name, image):
        """Save Image to directory."""
        cv.imwrite(image_name, image)

    def image_resize(self, image):
        """Resize Image."""
        return cv.resize(image, RESIZE_WIDTH_HEIGHT)

    def crop_image(self, image, squares):
        """
        Crop Images around square and save.
        Input: image and squares point
        Output: Cropped Image
        """
        x, y, width, height = cv.boundingRect(squares[0])
        cropped_image = image[y:y+height, x:x+width]
        # resize cropped image
        cropped_image = self.image_resize(cropped_image)
        return cropped_image

    def detect_square_shape(self, image_name):
        """
        Detect square in image and draw it.
        Input: Path to input Images
        Output: Save squared image
        """
        image = cv.imread(image_name)
        # # images shape
        # height, width, _ = image.shape

        # orig_image = image.copy()

        # print("image_name: ", image_name)
        # find square in image
        squares = self.find_contours(image)
        # draw Contours on image if sqaure points presents
        if squares:
            # draw square around the tiles
            # image = self.draw_contours(image, squares)

            # Crop The images around squares
            image = self.crop_image(image, squares)

            # get image name
            image_filename = image_name.split("/")[-1]
            # # store cropped images in separate video dir name
            # video_dir_name = image_name.split("/")[-2]
            # image_name = ("%s/%s/%s" % (
            #     CROPPED_IMAGES_PATH, video_dir_name, image_filename))
            image_name = ("%s/%s" % (
                CROPPED_IMAGES_PATH, image_filename))

            # save image to output dir
            self.save_image(image_name, image)

        # display image
        # self.display_image(image)

    def draw_contours(self, image, squares):
        """
        Draw Square on image.
        Input: image, squares
        Output: image
        """
        thickness = 0
        _ = cv.drawContours(image, squares, -1, (255, 0, 0),
                            thickness, cv.LINE_AA)
        return image

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

    def find_contours(self, image):
        """
        Identify contour in Image.
        Input: image
        Output: squares points
        """
        squares = []
        # convert the image to grayscale, blur it slightly,
        # and threshold it

        # For better accuracy, use binary images. So before finding contours,
        # apply threshold or canny edge detection.
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv2.bilateralFilter() is highly effective in noise removal
        # while keeping edges sharp
        # img = cv.bilateralFilter(gray, 11, 17, 17)

        img = cv.GaussianBlur(gray, (5, 5), 0)

        for gray in cv.split(img):
            for thrs in range(0, 255, 26):
                if thrs == 0:
                    thresh = cv.Canny(gray, 0, 50, apertureSize=5)
                    thresh = cv.dilate(thresh, None)
                else:
                    retval, thresh = cv.threshold(
                        gray, thrs, 255, cv.THRESH_BINARY)

                # perform a series of erosions and dilations to remove
                # any small blobs of noise from the thresholded image
                thresh = cv.erode(thresh, None, iterations=2)
                thresh = cv.dilate(thresh, None, iterations=4)

                contours, hierarchy = cv.findContours(
                    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    # area of contour
                    area = cv.contourArea(cnt)
                    # If area not within range, discard image
                    if area < MIN_AREA_512 or area > MAX_AREA_512:
                        continue
                    # print("before area: ", area)

                    cnt_len = cv.arcLength(cnt, True)
                    cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                    # area of contour
                    area = cv.contourArea(cnt)
                    # print("After area: ", area)
                    # print(len(cnt))

                    if len(cnt) == 4 and area > MIN_AREA_512 and \
                            area < MAX_AREA_512 and cv.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max(
                            [self.angle_cos(
                                cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4])
                                for i in range(4)])
                        # print("MAX_COS: ", max_cos)
                        if max_cos < MAX_COS:
                            squares.append(cnt)

        return squares


def help():
    print("Help: This program detect Square shape tile in images, "
          "If tile detected, crop the tile image and save to "
          "cropped directory path: %s\n" % (CROPPED_IMAGES_PATH))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images-path',
                        help="images directory path")
    args = parser.parse_args()
    images_path = args.images_path
    if not images_path:
        help()
        print("usage: %s [-h] [--images-path IMAGES]" % (sys.argv[0]))
        return

    obj = ShapeDetector()
    obj.detect(images_path)
    print("Successfully cropped images into directory: ", CROPPED_IMAGES_PATH)


if __name__ == '__main__':
    main()
