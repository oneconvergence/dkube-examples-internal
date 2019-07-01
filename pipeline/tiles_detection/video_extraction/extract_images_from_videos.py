#!/usr/bin/python3

# Extract images from videos and save to extracted_images directory

import argparse
import os
import sys

import cv2 as cv

VIDEO_FILE_EXTENSION = ".MTS"
# output directory to store extracted images
OUTPUT_IMAGES_PATH = "./extracted_images"
# RESIZE_WIDTH_HEIGHT = (1080, 1080)  # square image
RESIZE_WIDTH_HEIGHT = (512, 512)  # square image


class VideoExtraction:
    """
    Extract Frames from videos.
    Input: Video Directory Path
    Output: Save images to extracted images directory
    """
    def __init__(self):
        # create output directory to store extracted images
        if not os.path.exists(OUTPUT_IMAGES_PATH):
            try:
                os.makedirs(OUTPUT_IMAGES_PATH)
            except FileExistsError:
                pass

    def image_resize(self, image):
        """Resize Image and save back."""
        return cv.resize(image, RESIZE_WIDTH_HEIGHT)

    def FrameCapture(self, video_path):
        # Path to video file
        for file in os.listdir(video_path):
            if file.endswith(VIDEO_FILE_EXTENSION):
                video_file = "{}/{}".format(video_path, file)
                cap = cv.VideoCapture(video_file)

                if not cap.isOpened():
                    print("Failed to open file %s" % (video_file))
                    sys.exit()

                frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv.CAP_PROP_FPS)
                frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)

                print("frame_width: %s\nframe_height: %s\nframe_rate: %s\n"
                      "frame_count: %s" % (frame_width, frame_height, fps,
                                           frame_count))

                print("Extracting Frames from Video: %s ..." % (video_file))
                for frameNum in range(0, int(frame_count)):
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    # if frame is read correctly ret is True
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    # resize image
                    # original image size: 1920 * 1080 pixels (600kB)
                    # frame = self.image_resize(frame)

                    name = ("%s_tile_%s.jpg" % (file, frameNum))
                    # print(name)
                    cv.imwrite("%s/%s" % (OUTPUT_IMAGES_PATH, name), frame)

                print("Finished Extracting Frames from Video: %s ." % (
                    video_file))


def help():
    print("Help: This program extract frames from videos "
          "and save to output directory path: %s\n" % (OUTPUT_IMAGES_PATH))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', help="video directory path")
    args = parser.parse_args()
    video_dir = args.video_dir
    if not video_dir:
        help()
        print("usage: %s [-h] [--video-dir VIDEO_DIR]" % (sys.argv[0]))
        return

    obj = VideoExtraction()
    obj.FrameCapture(video_dir)


if __name__ == '__main__':
    main()
