import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


class LoadResizeImages():
    """
    This class loads and processes (scales & resizes) the content & style images 
    for the Neural Style Transfer.
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
    

class HubStylizedImage():
    """Use transfer learning to make a stylized image quickly."""
    def __init__(self, 
                 content_image, 
                 style_image, 
                 module_to_load='https://tfhub.dev/google/magenta/'
                     'arbitrary-image-stylization-v1-256/1'):
        self.content_image = content_image
        self.style_image = style_image
        self.module_to_load = hub.load(module_to_load)
        
    def get_stylized_image(self):
        tensor = self.module_to_load(tf.constant(self.content_image),
                                     tf.constant(self.style_image))[0]
        self.stylized_image = self._convert_to_image(tensor)
        return self.stylized_image
        
        
    def save_image(self, path, resized_width=None, resized_height=None):
        """
        Save image to specified path.
        Default sizes: width=384, height=512
        """
        resized_image = self.stylized_image.resize(
            size=(resized_width, resized_height))
        resized_image.save(path)
    
    def _convert_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
