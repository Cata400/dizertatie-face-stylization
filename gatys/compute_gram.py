import os
from utils import *
from model import StyleContentModel

gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)


styles = {
    'celeba': '/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/',
    'aahq': '/home/catalin/Desktop/Disertatie/Datasets/aahq/aligned_used/',
    'sketches': '/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/'
    }


results_dir = '../../Results/'
result_file = '../../Metrics/gram_features_normalized_results.txt'
f = open(result_file, 'w')


def gram_score(gen_path, style_path, vgg_model):
    gen_image = load_image(gen_path)
    style_image = load_image(style_path)
        
    gen_features = vgg_model(gen_image)
    target_features = vgg_model(style_image)
    
    for key in gen_features["style"].keys():
        continue
        # gen_features["style"][key] = (gen_features["style"][key] - tf.reduce_min(gen_features["style"][key])) / (tf.reduce_max(gen_features["style"][key]) - tf.reduce_min(gen_features["style"][key]))
        # target_features["style"][key] = (target_features["style"][key] - tf.reduce_min(target_features["style"][key])) / (tf.reduce_max(target_features["style"][key]) - tf.reduce_min(target_features["style"][key]))
    
    gram = style_loss(gen_features, target_features)
    
    return gram.numpy()
    


if __name__ == '__main__':
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']


    vgg_model = StyleContentModel(content_layers, style_layers)
    
    # gen_path = '/home/catalin/Desktop/Disertatie/Results/Gatys_ffhq_sketches_1k/00000+img99.png'
    # style_path = '/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/img99.png'
    # gram_score = gram_score(gen_path, style_path, vgg_model)
    # print(gram_score)


    for directory in sorted(os.listdir(results_dir)):
        if 'ffhq' in directory and 'old' not in directory:
            style = directory.split('_')[2]
            
            print(f'Calculating Gram for {directory} and {styles[style]} ...')
            print(f'Calculating Gram for {directory} and {styles[style]} ...', file=f)
            
            gram_scores = []
            
            for i, image in enumerate(sorted(os.listdir(os.path.join(results_dir, directory)))):
                style_image = image.split('+')[1]
                
                if 'JojoGAN' in directory:
                    style_image = "_".join(style_image.split('_')[:-2]) + '.png'
                
                if style == 'celeba':
                    style_image = style_image.replace('png', 'jpg')
            
                gen_path = os.path.join(results_dir, directory, image)
                style_path = os.path.join(styles[style], style_image)
                
                gram = gram_score(gen_path, style_path, vgg_model)
                gram_scores.append(gram)
                
            avg_gram = sum(gram_scores) / len(gram_scores)
            
            print(f'Average ID for {directory}: {avg_gram}')
            print(f'Average ID for {directory}: {avg_gram}', file=f)
            
            print()
            print("", file=f)