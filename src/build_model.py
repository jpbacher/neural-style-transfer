"""
This module contains classes & functions to run Neural Style Transfer.
"""

import time
import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub


class ContentAndStyleImage():
    """
    This class loads and processes (resize, scales) the content & style images 
    for the Neural Style Transfer.
    Arguments:
    
    Returns:
    
    """
    def __init__(self, content_path, style_path, resolution_max):
        self.content_path = content_path
        self.style_path = style_path
        self.content_img = self._get_image(content_path, resolution_max)
        self.style_img = self._get_image(style_path, resolution_max)
    
    def _get_image(self, path, resolution_max):
        image = mpl.image.imread(path)
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        dimension_max = resolution_max
        shape = tf.cast(tf.shape(image)[:-1], dtype=tf.float32)
        long_dimension = max(shape)
        scale = dimension_max / long_dimension
        new_shape = tf.cast(shape*scale, tf.int32)
        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        return image
    
    def show_image(self, image):
        """Plot an image."""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        plt.imshow(image)
        
    def rotate_image(self, image, rotations):
        """Rotate image 90 degrees counter-clockwise."""
        return tf.image.rot90(image, k=rotations)

