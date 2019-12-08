import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub


def resize_image(image):
    max_dimension = 512
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
    shape = tf.cast(tf.shape(image)[:-1], dtype=tf.float32)
    long_dimension = max(shape)
    scale = max_dimension / long_dimension
    
    new_shape = tf.cast(shape*scale, tf.int32)
    
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image

def show_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)