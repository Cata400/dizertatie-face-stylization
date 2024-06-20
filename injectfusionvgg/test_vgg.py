import torch
import torch.nn as nn
from torchvision.models import vgg19
from PIL import Image
from torchvision import transforms



image = Image.open('../Images/Gatys/Content/labrador.jpg')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0)
vgg_model = vgg19(weights='DEFAULT')

with torch.no_grad():
    y = vgg_model(image)
    y = nn.functional.softmax(y, dim=1)
    
    # Print top 5 classes predicted by the model
    print(y[0].topk(5))
    

content_model_vgg = nn.Sequential()
content_layer = 'block5_conv2'

i = 0  # increment every time we see a conv
j = 1  # increment every time we see a maxpool
for layer in vgg_model.get_submodule("features").children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f'block{j}_conv{i}'
    elif isinstance(layer, nn.ReLU):
        name = f'block{j}_relu{i}'
    elif isinstance(layer, nn.MaxPool2d):
        name = f'block{j}_pool{i}'
        j += 1
        i = 0
    elif isinstance(layer, nn.BatchNorm2d):
        name = f'block{j}_bn{i}'
    else:
        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    content_model_vgg.add_module(name, layer)

    if name == content_layer:
        break
    
    
features = content_model_vgg(image)
print(features.shape)
