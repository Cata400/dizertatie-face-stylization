import os
import tensorflow as tf
import time
import random

from utils import *
from model import StyleContentModel

import argparse


gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)

seed = 42
tf.random.set_seed(seed)
random.seed(seed)


def main(content_path, style_path, output_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    
    content_weight = 1e3
    style_weight = 1e-2
    epochs = 1000

    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']


    extractor = StyleContentModel(content_layers, style_layers)
    
    targets = {
        'content': extractor(content_image)['content'],
        'style': extractor(style_image)['style']
    }

    image = tf.Variable(content_image)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    start = time.time()
    for _ in range(epochs):
        train_step(image, extractor, optimizer, targets, content_weight, style_weight)
    
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
        
    output_image = tensor_to_image(image)
    output_image.save(output_path)
        

if __name__ == '__main__':
    # content_path =  os.path.join('..', '..', 'Images', 'Gatys', 'Content', "labrador.jpg")
    # style_path =  os.path.join('..', '..', 'Images', 'Gatys', 'Style', "stary_night.jpg")
    
    parser = argparse.ArgumentParser(description="Neural Style Tranfer")
    
    parser.add_argument('content', type=str, help='Content image path')
    parser.add_argument('style', type=str, help='Style image path')
    parser.add_argument('output', type=str, help='Output image path')
    
    args = parser.parse_args()
    
    main(args.content, args.style, args.output)

