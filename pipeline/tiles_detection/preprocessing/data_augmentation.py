#!/usr/bin/python3

"""
Data Augmentation to make balanced dataset.
1. Flip the images.
2. Ligthing the images.
"""

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import cv2 as cv

IMAGE_SIZE = 299
DATA_AUGMENTED_RESIZE_DIR = "./augmented_images/resized_images/"
DATA_AUGMENTED_FLIP_DIR = "./augmented_images/flip_images/"
DATA_AUGMENTED_LIGHTING_DIR = "./augmented_images/lighting_images/"


class DataAugmentation:

    # Collect image file paths
    def resize_images(self, image_dir_path, resized_images_path):
        """
        Resize Image to IMAGE_SIZE and store to separate
        resize directory.
        """
        for file in os.listdir(image_dir_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_name = "{}/{}".format(image_dir_path, file)
                resized_image = self.tf_augmentation(image_name, resize=True)
                # save resize image to new image path
                new_image = ("{}/DA_RESIZE_{}".format(
                    resized_images_path, file))
                self.display_and_save_image(resized_image, new_image)

    def tf_augmentation(self, image_name, resize=False, flip_left_right=False,
                        flip_up_down=False, flip_transpose=False):
        # Inserts a placeholder for a tensor that will be always fed.
        filename = tf.placeholder(tf.string, name="inputFile")
        # Use feed_dict to feed values to TensorFlow placeholders
        feed_dict = {filename: image_name}
        # Reads and outputs the entire contents of the input filename.
        # filename: A Tensor of type string.
        fileContent = tf.read_file(filename, name="loadFile")
        # Decode a JPEG-encoded image to a uint8 tensor.
        image = tf.image.decode_jpeg(fileContent, name="decodeJpeg")

        if resize:
            # Resize images to size using nearest neighbor interpolation.
            output_image = tf.image.resize_images(
                image, size=[IMAGE_SIZE, IMAGE_SIZE],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif flip_left_right:
            # flip image left_right
            output_image = tf.image.flip_left_right(image)
        elif flip_up_down:
            # flip image up_down
            output_image = tf.image.flip_up_down(image)
        elif flip_transpose:
            # flip image transpose
            output_image = tf.image.transpose_image(image)

        with tf.Session().as_default():
            # actualImage = image.eval(feed_dict)
            output_image = output_image.eval(feed_dict)
        return output_image

    def flip_images(self, image_dir_path, flip_images_path):
        """Flip Resized images and save to new flip directory."""
        # print("inside flip: image_dir_path: ", image_dir_path)
        for file in os.listdir(image_dir_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_name = "{}/{}".format(image_dir_path, file)
                # # flip image left_right
                flag = "left_right"
                flipped_image = self.tf_augmentation(
                    image_name, flip_left_right=True)
                # save flipped image to new image path
                new_image = ("{}/DA_flip_{}_{}".format(
                    flip_images_path, flag, file))
                self.display_and_save_image(flipped_image, new_image)

                # # flip image up_down
                flag = "up_down"
                flipped_image = self.tf_augmentation(
                    image_name, flip_up_down=True)
                # save flipped image to new image path
                new_image = ("{}/DA_flip_{}_{}".format(
                    flip_images_path, flag, file))
                self.display_and_save_image(flipped_image, new_image)

                # # flip image transpose_image
                # flag = "transpose"
                # flipped_image = self.tf_augmentation(
                #     image_name, flip_transpose=True)
                # # save flipped image to new image path
                # new_image = ("{}/DA_flip_{}_{}".format(
                #     flip_images_path, flag, file))
                # self.display_and_save_image(flipped_image, new_image)

    def lighting_images(self, image_dir_path, lighting_images_path):
        """Lighting images to shaded image using OpenCV."""
        # Gaussian distribution parameters
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        for file in os.listdir(image_dir_path):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                image_name = "{}/{}".format(image_dir_path, file)
                image = cv.imread(image_name)
                row, col, _ = image.shape
                print(row, col)

                # gaussian = np.random.random((row, col, 1)).astype(np.float32)
                gaussian = np.random.normal(mean, sigma, (row, col))
                # gaussian = np.concatenate(
                #     (gaussian, gaussian, gaussian), axis=2)
                gaussian = np.column_stack((gaussian, gaussian, gaussian))
                gaussian_img = cv.addWeighted(
                    image, 0.75, 0.25 * gaussian, 0.25, 0)

                cv.imshow("resized_image", gaussian_img)
                cv.waitKey(0)
                break
                # save flipped image to new image path
                # new_image = ("{}/DA_lighting_{}".format(
                #     lighting_images_path, file))
                # self.display_and_save_image(flipped_image, new_image)

    def display_and_save_image(self, resized_image, new_image):
        # show using opencv
        resized_image = cv.cvtColor(resized_image, cv.COLOR_RGB2BGR)
        # cv.imshow("resized_image", resized_image)
        # cv.waitKey(0)
        cv.imwrite(new_image, resized_image)


def help():
    print("Help: This program does data augmentation to make dataset "
          "balanced.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path',
                        help="images directory path")
    parser.add_argument('--flip-images', action="store_true",
                        help="flip images left_right or up_down")
    # parser.add_argument('--lighting', action="store_true",
    #                     help="add lighting shade to image")

    args = parser.parse_args()
    images_path = args.images_path
    flip_images_flag = args.flip_images
    # lighting_flag = args.lighting

    if not images_path:
        help()
        print("usage: %s [-h] [--images-path IMAGES_PATH] "
              "[--flip-images" % (sys.argv[0]))
        return

    da_obj = DataAugmentation()

    # Resize all images first
    resized_images_path = (images_path + "/" + DATA_AUGMENTED_RESIZE_DIR)
    print("Resizing images ...")
    if not os.path.exists(resized_images_path):
        try:
            os.makedirs(resized_images_path)
        except FileExistsError:
            pass
        da_obj.resize_images(images_path, resized_images_path)

    print("Resizing images Done!!!")

    # FLIP Resized Images
    if flip_images_flag:
        print("Flipping Resized Images ...")
        # resized images already exists
        flip_images_path = (images_path + "/" + DATA_AUGMENTED_FLIP_DIR)
        if not os.path.exists(flip_images_path):
            try:
                os.makedirs(flip_images_path)
            except FileExistsError:
                pass
        da_obj.flip_images(resized_images_path, flip_images_path)
        print("Flipping Resized Done!!!")

    # # Lighting condition
    # if lighting_flag:
    #     print("Lighting images ...")
    #     # resized images already exists
    #     lighting_images_path = (images_path + "/" +
    #                             DATA_AUGMENTED_LIGHTING_DIR)
    #     if not os.path.exists(lighting_images_path):
    #         try:
    #             os.makedirs(lighting_images_path)
    #         except FileExistsError:
    #             pass
    #     da_obj.lighting_images(resized_images_path, lighting_images_path)
    #     print("Lighting images Done!!!")


if __name__ == '__main__':
    main()
