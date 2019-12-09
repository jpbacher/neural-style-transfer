import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub


class ProcessImages():
    """
    This class loads and processes (scales & resizes) the content & style images 
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
    

class NST():
    """ """
    def __init__(self):
        # intermediate layers to represent content & style of image
        self.content_layers = ['block5_conv2']
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]
        self.content_layer_count = len(self.content_layers)
        self.style_layer_count = len(self.style_layers)
        self._build_vgg_model()
        
    def _build_vgg_model(self, layer_names):
        """
        This loads a VGG19 model, and accesses the intermediate layers.
        """
        # load pretrained VGG19 model.
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        self.content_outputs = [self.vgg.get_layer(
            name).output for name in self.content_layers]
        self.style_outputs = [self.vgg.get_layer(
            name).output for name in self.style_layers]
        self.model_outputs = self.content_outputs + self.style_outputs
        self.model = tf.keras.Model([self.vgg.input], self.model_outputs)
        
    def _get_gram_matrix(self, input_tensor):
        """
        Compute the gram matrix of the input tensor.
        """
        gm_result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return gm_result / num_locations
    
    def _get_total_loss(self, outputs):
        content_outputs = outputs['content']
        style_outputs = output['style']
        style_loss = tf.add_n([tf.reduce_mean(
            (style_outputs[name] - style_targets[name])**2) 
            for name in style_outputs.keys()])
        style_loss *= style_weight / self.style_layer_count
        content_loss = tf.add_n(([tf.reduce_mean(
            (content_outputs[name] - content_targets[name])**2)
            for name in content_outputs.keys()])
   
    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as t:
            outputs = extractor(image)
            loss = self._get_total_loss(outputs)
        gradients = t.gradient(loss, image)
        optimizer.apply_gradients([(gradients, image)])
        image.assign(clip(image))