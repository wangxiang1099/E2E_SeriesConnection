import numpy as np 
import cv2
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor(object):
    def __call__(self, image):

        image = F.to_tensor(image)
        return image

class Normalize(object):
    def __call__(self, image):
        image = F.normalize(image,[0.5,0.5,0.5],[0.5,0.5,0.5])
        return image

def get_image_encoding_transforms():

    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize())
    return Compose(transforms)