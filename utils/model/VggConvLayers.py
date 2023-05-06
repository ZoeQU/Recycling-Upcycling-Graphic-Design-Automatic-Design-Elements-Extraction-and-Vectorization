# -*- coding:utf-8 -*-
from torchvision import models
import torch.utils.model_zoo as model_zoo
import setting
from torch import nn
import cv2
import torch
from torchvision import models, transforms
from model.vgg import VGG16

preprocess_transform = transforms.Compose([transforms.ToTensor()])


class Vgg16Layers(models.VGG):
    def __init__(self):
        super(Vgg16Layers, self).__init__()

    def forward(self, x):
        layers_outputs = []
        for l in self.features:
            x = l(x)
            if isinstance(l, nn.Conv2d):
                layers_outputs.append(x)
        x = self.avgpool(x)
        layers_outputs.append(x)
        x = torch.flatten(x, 1)

        fclinear = []
        for l in self.classifier:
            x = l(x)
            fclinear.append(x)
        for j in range(len(fclinear) - 1):
            layers_outputs.append(fclinear[j])
        return layers_outputs

def vgg16_conv_layers():
    model = Vgg16Layers()
    # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    checkpoint = torch.load("../params/vgg16-397923af.pth")
    model.load_state_dict(checkpoint)
    return model

"""test"""
def feature_extraction(image):
    """vgg16 feature extraction"""
    # image = load_image(image_path)
    dev = torch.device("cuda")
    image = preprocess_transform(image).unsqueeze(0).to(dev)
    image_size = image.squeeze().shape
    image_size = tuple([image_size[1], image_size[2], image_size[0]])
    dev = torch.device("cuda")
    model = vgg16_conv_layers()
    model.to(dev)
    features = model(image)
    # feature = features[-1].data.cpu().numpy()  #np.array.size:9216
    feature = features[-1]  # torch.Size:[1, 4096]
    return feature

img = cv2.imread('/home/user/0-zoe_project/ImgVectorization/doc/input_cropedpattern/patterns37_cropped.png')
f = feature_extraction(img)