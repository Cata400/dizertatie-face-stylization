from utils import *


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, content_layers, style_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = self.get_intermediary_outputs(content_layers + style_layers)
        
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        
        self.vgg.trainable = False


    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        
        content_outputs, style_outputs = outputs[:self.num_content_layers], outputs[self.num_content_layers:]

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
    
    
    def get_intermediary_outputs(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model
    
    
    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        
        return result / (num_locations)