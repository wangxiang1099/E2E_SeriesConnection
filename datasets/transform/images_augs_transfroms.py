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

import cv2
import numpy as np
from torchvision.transforms import functional as F

class Resize(object):

    def __init__(self):
        
        self.out_width = 768
        self.out_height = 768

    def __call__(self, img, bboxes):

        h, w = img.shape[0:2]

        h_ratio = self.out_height/h
        w_ratio = self.out_width/w

        resized_image = cv2.resize(img,(self.out_width,self.out_height))
        
        bboxes[:,:,0] = bboxes[:,:,0]*w_ratio
        bboxes[:,:,1] = bboxes[:,:,1]*h_ratio
        
        return resized_image, bboxes


class RandomScale(object):
    min_size = 768

    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img, bboxes):

        min_size = self.min_size
        h, w = img.shape[0:2]

        scale = 1.0
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
        random_scale = np.array([0.5, 1.0, 1.5, 2.0])
        scale1 = np.random.choice(random_scale)
        if min(h, w) * scale * scale1 <= min_size:
            scale = (min_size + 10) * 1.0 / min(h, w)
        else:
            scale = scale * scale1
        bboxes *= scale
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img, bboxes

def image_augs():

    transforms = []
    transforms.append(Resize())

    return Compose(transforms)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image, bboxes)
        return image, bboxes

def padding_image(image, imgsize):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    #### IC15 for 0.6, MLT for 0.35 #####
    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

"""
         random_transforms = [image]
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        random_transforms = random_horizontal_flip(random_transforms)
        random_transforms = random_rotate(random_transforms)
"""


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
