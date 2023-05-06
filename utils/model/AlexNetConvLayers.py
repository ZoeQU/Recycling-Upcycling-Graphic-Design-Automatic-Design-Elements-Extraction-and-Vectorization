from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch import nn
import torch

class AlexNetConvLayers(models.AlexNet):

    def __init__(self):
        super(AlexNetConvLayers, self).__init__()

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
        for j in range(2):
            layers_outputs.append(fclinear[j])
        return layers_outputs


def alexnet_conv_layers():
    model = AlexNetConvLayers()
    #model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    checkpoint = torch.load("utils/params/alexnet-owt-4df8aa71.pth")
    model.load_state_dict(checkpoint)
    return model
