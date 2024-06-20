import numpy as np
import tensorflow as tf
from PIL import Image


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        
    return Image.fromarray(tensor)


def load_image(path):
    max_dim = 512
    
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    
    return img


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, targets, content_weight, style_weight):
    content_outputs = outputs['content']
    style_outputs = outputs['style']
    
    content_targets = targets['content']
    style_targets = targets['style']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                            for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                                for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    
    loss = style_loss + content_loss
    return loss


def style_loss(outputs, targets):
    style_outputs = outputs['style']
    style_targets = targets['style']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                            for name in style_outputs.keys()])
    style_loss *= 1 / len(style_outputs)
    
    return style_loss


@tf.function()
def train_step(image, extractor, optimizer, targets, content_weight, style_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, targets, content_weight, style_weight)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_image(image))