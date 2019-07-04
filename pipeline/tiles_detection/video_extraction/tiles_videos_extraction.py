#!/usr/bin/python3

"""
Extract frames from videos, detect tile in images.
If Tile detected, crop tile image and save image to
CROPPED_IMAGES_PATH
"""


import argparse
import os
import sys
import time
import multiprocessing

import cv2 as cv

# from threading import Thread
from video_extraction.shape_detector import ShapeDetector

VIDEO_FILE_EXTENSION = ".MTS"
# output directory to store cropped images
CROPPED_IMAGES_PATH = "./preprocessed_data" + "/data"
# output directory to store extracted images
EXTRACTED_IMAGES_PATH = "./extracted_images"
# RESIZE_WIDTH_HEIGHT = (1080, 1080)  # square image
RESIZE_WIDTH_HEIGHT = (512, 512)  # square image
# framerate to reduce FPS, default FPS is 50
FRAMERATE = 20


class VideoExtraction:
    """
    Extract Frames from videos.
    Input: Video Directory Path
    Output: Save images to extracted images directory
    """
    def __init__(self):
        # create output directory to store extracted images
        if not os.path.exists(EXTRACTED_IMAGES_PATH):
            try:
                os.makedirs(EXTRACTED_IMAGES_PATH)
            except FileExistsError:
                pass
        # video capture
        self.cap = None
        # video dir path
        self.video_path = None
        # obj of shape detector
        self.sd = ShapeDetector()

    def image_resize(self, image):
        """Resize Image and save back."""
        return cv.resize(image, RESIZE_WIDTH_HEIGHT)

    def save_image(self, image_name, frame):
        """Save Image to directory."""
        cv.imwrite(image_name, frame)

    def video_extraction(self, video_filename):
        # store extracted images into separate directory name as
        # video_filename under EXTRACTED_IMAGES_PATH
        video_extracted_images_path = EXTRACTED_IMAGES_PATH + "/{}".format(
            video_filename)
        if not os.path.exists(video_extracted_images_path):
            try:
                os.makedirs(video_extracted_images_path)
            except FileExistsError:
                pass

        frame_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        frame_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv.CAP_PROP_FPS)
        frame_count = self.cap.get(cv.CAP_PROP_FRAME_COUNT)

        print("{0} {1} {0}".format("#"*10, video_filename))
        print("frame_width: %s\nframe_height: %s\nfps: %s\n"
              "frame_count: %s\nframe_rate: %s" % (
                frame_width, frame_height, fps,
                frame_count, FRAMERATE))

        # choose alternate frames, it will reduce FPS to half of
        # the original 50 // 20 = 2, then select frames with % 2 != 0
        if FRAMERATE >= fps:
            # if fps is same as framerate, then select all frames
            divider = 1
        else:
            # take frames with modulo != 0
            divider = fps // FRAMERATE
        # print("divider: ", divider)

        for frameNum in range(0, int(frame_count)):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # choose alternate frames
            if frameNum % divider != 0:
                continue

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # resize image
            # original image size: 1920 * 1080 pixels (600kB)
            frame = self.image_resize(frame)

            name = ("%s/%s_tile_%s.jpg" % (
                video_extracted_images_path, video_filename, frameNum))
            # print(name)

            # save frames to directory
            self.save_image(name, frame)

            # # individual detect shapes in image
            # # detect square shape in tile images
            # self.sd.detect_square_shape(name)

        # detect square in tile image and do some preprocessing
        # using multiprocessing
        print("ShapeDetector for : {}".format(video_extracted_images_path))
        self.sd.detect(video_extracted_images_path)

    def start(self, video_filename):
        """
        Multithreading approach to get frames from video and
        identify tiles in frames.
        """
        # t1 = Thread(target=self.video_extraction, args=(video_filename,))
        # t1.start()
        # t1.join()

        # without thread
        self.video_extraction(video_filename)

    def video_processing(self, video_filename):
        if video_filename.endswith(VIDEO_FILE_EXTENSION):
            video_file = "{}/{}".format(self.video_path, video_filename)

            # Capture Frames from video file
            self.cap = cv.VideoCapture(video_file)

            if not self.cap.isOpened():
                print("Failed to open file %s" % (video_file))
                sys.exit()

            print("Extracting Frames from Video: %s and "
                  "identifying tiles in the frames ..." % (video_file))

            # extract frames from video and identify tile in frames
            self.start(video_filename)

            print("Finished Extracting Frames from Video: %s ." % (
                video_file))

    def FrameCapture(self, video_path, flag=False):
        """
        Capture frames from videos, identify tiles images from frames.
        Store cropped images.
        Input: Path to videos directory, flag: to store extracted images
        Output: Cropped images
        """
        try:
            self.video_path = video_path
            # multiprocessing inside multiprocessing (shape detector also uses)
            # gives error: daemonic processes are not allowed to have children

            with multiprocessing.Pool() as pool:
                pool.map(self.video_processing, os.listdir(video_path))

        except Exception as err:
            print("Unable to extract frames from videos. Reason: "
                  "{}".format(err))
        finally:
            # if flag True, then keep extracted images else delete
            # extracted images
            if flag is False:
                cmd = ("rm -rf {}".format(EXTRACTED_IMAGES_PATH))
                os.system(cmd)


# time decorator
def timeit(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time()
        print("Total time taken: {:.3f} sec".format(t2 - t1))
    return wrapper


def help():
    print("Help: This program extract frames from videos, "
          "detect tile in frames.\n"
          "If tile detected, crop the tile image and save to "
          "output directory path: %s\n" % (CROPPED_IMAGES_PATH))


# @timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', help="video directory path")
    parser.add_argument('--store-extracted-images',
                        action="store_true",
                        help=("Store extracted images from videos or not. "
                              "Default: False"))
    args = parser.parse_args()
    video_dir = args.video_dir
    store_extracted_images = args.store_extracted_images
    if not video_dir:
        help()
        print("usage: %s [-h] [--video-dir VIDEO_DIR]" % (sys.argv[0]))
        return

    obj = VideoExtraction()
    obj.FrameCapture(video_dir, store_extracted_images)
    print("Successfully extracted images from videos. "
          "Cropped images are stored in directory: ", CROPPED_IMAGES_PATH)


if __name__ == '__main__':
    main()
