import random
import torch
import numpy as np 
import cv2
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):

        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __call__(self, image, target):
        image = F.normalize(image,[0.5,0.5,0.5],[0.5,0.5,0.5])
        return image, target


def get_transforms_recognition():

    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize())
    return Compose(transforms)

if  __name__ == "__main__":

    import sys 
    sys.path.append("/home/wx/project/now_project/iocr_trainning")
    
    from datasets.invoices.synthesisInvoices import SynthesisInvoices

    dataset = SynthesisInvoices(
        root= "/home/wx/data/iocr_training/recognition_data", splits = ["rec1_train","rec2_train"], 
        transforms=get_transforms_recognition(),
    )
    print(dataset[2][1])

    from datasets.ocr.invoices.synthesisInvoices import SynthesisInvoices
