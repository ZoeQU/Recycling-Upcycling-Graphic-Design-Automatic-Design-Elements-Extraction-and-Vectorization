# -*- coding:utf-8 -*-
import numpy as np
import heapq
import cv2
import torch
from torchvision import models, transforms
from utils.model.AlexNetConvLayers import alexnet_conv_layers

from PIL import Image, ImageDraw, ImageFilter

preprocess_transform = transforms.Compose([transforms.ToTensor()])


def load_image(img_path):
    dev = torch.device("cuda")
    image = Image.open(img_path).convert('RGB')
    # image = image.resize((227,227))
    return preprocess_transform(image).unsqueeze(0).to(dev)

def feature_extraction(image):
    """Alexnet feature extraction"""
    # image = load_image(image_path)
    dev = torch.device("cuda")
    image = preprocess_transform(image).unsqueeze(0).to(dev)
    image_size = image.squeeze().shape
    image_size = tuple([image_size[1], image_size[2], image_size[0]])
    dev = torch.device("cuda")
    model = alexnet_conv_layers()
    model.to(dev)
    features = model(image)
    # feature = features[-1].data.cpu().numpy()  #np.array.size:9216
    feature = features[-1]  # torch.Size:[1, 4096]

    return feature


def CNNfeaturesSim(img1, img2):
    feature1 = feature_extraction(img1)
    feature2 = feature_extraction(img2)
    # cosine_similarity = torch.cosine_similarity(feature1, feature2, dim=1).data.cpu().numpy().astype(float)
    # cos_sim = (cosine_similarity + 1) / 2
    L1_distance = torch.dist(feature1, feature2, p=1).data.cpu().numpy().astype(float)
    L1_sim = L1_distance / 4096
    # L2_distance = torch.dist(feature1, feature2, p=2).data.cpu().numpy().astype(float)
    # L2_sim = L2_distance / 9216
    return L1_sim