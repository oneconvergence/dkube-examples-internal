#!/usr/bin/python3

"""
Data Augmentation to make balanced dataset.
1. Resize the images.
2. Flip the images.
3. Ligthing the images.
"""

import os
import multiprocessing

import tensorflow as tf
import cv2

import params


class DataAugmentation:
    """Perform DataAugmentation all NIH Chest X-Ray images."""

    def resize_images(self):
        """with multiprocessing, Resize all the images."""
        with multiprocessing.Pool() as pool:
            pool.map(self._resize_image, os.listdir(os.path.join(
                params.DATA_FOLDER, "images")))

    def _resize_image(self, image):
        """Resize all the images."""
        image = os.path.join(params.DATA_FOLDER, "images", image)
        # resize image using tensorflow
        resized_image = self.tf_augmentation_resize(
            image)
        self.save_image(resized_image, image)

    def tf_augmentation_resize(self, image_name):
        """Resize image using tensorflow API."""
        # Inserts a placeholder for a tensor that will be always fed.
        filename = tf.compat.v1.placeholder(tf.string, name="inputFile")
        # Use feed_dict to feed values to TensorFlow placeholders
        feed_dict = {filename: image_name}
        # Reads and outputs the entire contents of the input filename.
        # filename: A Tensor of type string.
        fileContent = tf.io.read_file(filename, name="loadFile")
        # Decode a JPEG-encoded image to a uint8 tensor.
        image = tf.image.decode_png(fileContent)

        # Resize images to size using nearest neighbor interpolation.
        output_image = tf.image.resize(
            image, size=params.RESIZE_IMAGE,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # print("shape: ", tf.shape(output_image))
        with tf.compat.v1.Session().as_default():
            # actualImage = image.eval(feed_dict)
            output_image = output_image.eval(feed_dict)
        return output_image

    def horizontal_flip(self):
        """with multiprocessing, Horizontal flip all the images."""
        with multiprocessing.Pool() as pool:
            pool.map(self._horizontal_flip, os.listdir(os.path.join(
                params.DATA_FOLDER, "images")))

    def _horizontal_flip(self, image):
        """Flip an image horizontally (left to right)."""
        suffix = "h_flip"
        name, ext = os.path.splitext(image)
        new_image_name = ("{}-{}{}".format(name, suffix, ext))
        image = os.path.join(params.DATA_FOLDER, "images", image)
        new_image_name = os.path.join(
            params.DATA_FOLDER, "images", new_image_name)
        # resize image using tensorflow
        resized_image = self.tf_augmentation_flip(
            image)
        self.save_image(resized_image, new_image_name)

    # because of multiprocessing error, separate the tf functions
    def tf_augmentation_flip(self, image_name):
        """Flip an image horizontally (left to right)."""
        # Inserts a placeholder for a tensor that will be always fed.
        filename = tf.compat.v1.placeholder(tf.string, name="inputFile")
        # Use feed_dict to feed values to TensorFlow placeholders
        feed_dict = {filename: image_name}
        # Reads and outputs the entire contents of the input filename.
        # filename: A Tensor of type string.
        fileContent = tf.io.read_file(filename, name="loadFile")
        # Decode a JPEG-encoded image to a uint8 tensor.
        image = tf.image.decode_png(fileContent)

        # flip left to right
        output_image = tf.image.flip_left_right(image)

        with tf.compat.v1.Session().as_default():
            # actualImage = image.eval(feed_dict)
            output_image = output_image.eval(feed_dict)
        return output_image

    def random_brightness(self):
        """with multiprocessing, adjust random brightness of all the images."""
        with multiprocessing.Pool() as pool:
            pool.map(self._random_brightness, os.listdir(os.path.join(
                params.DATA_FOLDER, "images")))

    def _random_brightness(self, image):
        """Randomly adjust the brigthness(0.0, 32.0)."""
        suffix = "brigthness"
        name, ext = os.path.splitext(image)
        new_image_name = ("{}-{}{}".format(name, suffix, ext))
        image = os.path.join(params.DATA_FOLDER, "images", image)
        new_image_name = os.path.join(
            params.DATA_FOLDER, "images", new_image_name)
        print(new_image_name)
        # randomly adjust brightness bw (0.0, 32.0)
        resized_image = self.tf_augmentation_brightness(
            image)
        self.save_image(resized_image, new_image_name)

    # because of multiprocessing error, separate the tf functions
    def tf_augmentation_brightness(self, image_name):
        """Flip an image horizontally (left to right)."""
        # Inserts a placeholder for a tensor that will be always fed.
        filename = tf.compat.v1.placeholder(tf.string, name="inputFile")
        # Use feed_dict to feed values to TensorFlow placeholders
        feed_dict = {filename: image_name}
        # Reads and outputs the entire contents of the input filename.
        # filename: A Tensor of type string.
        fileContent = tf.io.read_file(filename, name="loadFile")
        # Decode a JPEG-encoded image to a uint8 tensor.
        image = tf.image.decode_png(fileContent)

        # Resize images to size using nearest neighbor interpolation.
        output_image = tf.image.random_brightness(image, 0.8)

        with tf.compat.v1.Session().as_default():
            # actualImage = image.eval(feed_dict)
            output_image = output_image.eval(feed_dict)
        return output_image

    @staticmethod
    def save_image(resized_image, new_image):
        # save image using OpenCV
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_image, resized_image)
