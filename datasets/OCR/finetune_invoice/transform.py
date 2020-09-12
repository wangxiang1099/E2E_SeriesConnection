import random
import torch
import numpy as np 
import cv2
from torchvision.transforms import functional as F
import pyclipper
from shapely.geometry import Polygon

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class maskToTensor(object):

    def __call__(self, image, target):
        
        target['detect_mask'] =  F.to_tensor(target['detect_mask']).squeeze(0)
        target['detect_border_mask'] =  F.to_tensor(target['detect_border_mask']).squeeze(0)       
        return image, target

class GtToTensor(object):

    def __call__(self, image, target):
        
        target['detect_gt'] =  F.to_tensor(target['detect_gt'])
        target['detect_gt_border'] =  F.to_tensor(target['detect_gt_border'])
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        
        return image, target

class ResizeBoundingBox(object):
    
    def __init__(self):
        self.resize_height = 1<<10
        self.resize_width = 1<<11

    def __call__(self, image, target):
            
        for k,v in target['anno_label']:
            pass
        return image, target

# use for get prepare data
class Resize(object):

    def __init__(self, out_width=480, out_height=32, keep_ratio = False,prepare=False):
        
        self.out_width = out_width
        self.out_height = out_height
        self.keep_ratio = keep_ratio
        self.prepare = prepare

    def __call__(self, image, target):
        
        print(target.keys())
        print(target['detect_gt'].shape)
        if not self.keep_ratio:
            resized = cv2.resize(image,(self.out_width,self.out_height))
            
            if self.prepare:
                resized_detect_gt = cv2.resize(target['detect_gt'],(self.out_width,self.out_height))
                resized_detect_gt_border = cv2.resize(target['detect_gt_border'],(self.out_width,self.out_height))
                resized_detect_border_mask = cv2.resize(target['detect_border_mask'],(self.out_width,self.out_height))
                resized_gt_dilate = cv2.resize(target['gt_dilate'],(self.out_width,self.out_height))

                h_ratio =  self.out_height / target['origin_h'] 
                w_ratio =  self.out_width / target['origin_w'] 
                
                print(h_ratio,w_ratio)
                for k,v in target['anno_label'].items():
                    box = v['boxes']
                    box[0] = int(box[0]*w_ratio)
                    box[2] = int(box[2]*w_ratio)

                    box[1] = int(box[1]*h_ratio)
                    box[3] = int(box[3]*h_ratio)
                    target['anno_label'][k]['boxes'] = box

        else:
            height, width, channel = image.shape
            ratio_h = self.out_height/height
            ratio_w = self.out_width/width
            ratio = min(ratio_h,ratio_w)

            resize_h = int(ratio*height)
            resize_w = int(ratio*width)

            image = cv2.resize(image,(resize_w,resize_h))
            resized = np.zeros((self.out_height, self.out_width, channel), dtype=np.float32)
            resized[:,:,:] = 127.
            resized[0:resize_h,0:resize_w,:] = image
        cv2.imwrite("/home/wx/tmp_pic/resized_test.jpg", resized)

        if self.prepare:
            target.update(detect_gt=resized_detect_gt,
                          detect_gt_border=resized_detect_gt_border,
                          detect_border_mask=resized_detect_border_mask,
                          gt_dilate=resized_gt_dilate, 
                          origin=resized.copy())

        return resized, target


# 用于 生成 gt_detect 和 gt_detect_border
class MakeSegDetectorData(object):

    def __init__(self, **kwargs):
        #self.min_text_size = 8
        self.shrink_ratio = 0.4

    def __call__(self, image, target, vis=True):
        '''
        data: a dict typically returned from `MakeICDARData`,
            where the following keys are contrains:
                image*, polygons*, ignore_tags*, shape, filename
                * means required.
        '''
        image_vis = image.copy()
        boxes = [t['boxes'] for t in target['anno_label'].values()]
        h, w = image.shape[0:2]
        print(h,w)
        gt = np.zeros((h, w, 1), dtype=np.float32)
        gt_dilate = np.zeros((h, w, 1), dtype=np.float32)

        polygons_all = []

        for box in boxes:
            
            x1,y1, x2,y2 = box

            polygons = ((x1,y1), (x2,y1),
                        (x2,y2), (x1,y2))

            polygons_all.append(polygons)

        mm = MakeBorderMap()
        canvas , mask = mm(image_vis, polygons_all)
        print(canvas.shape, mask.shape)

        for box in boxes:
            
            x1,y1, x2,y2 = box

            area = int(x2-x1)*int(y2-y1)
            longth = 2*int(x2-x1) + 2*int(y2-y1)

            distance = area * \
                    (1 - np.power(self.shrink_ratio, 2)) / longth
            
            shrinked_x1 = x1 + distance/2
            shrinked_y1 = y1 + distance
            shrinked_x2 = x2 - distance/2
            shrinked_y2 = y2 - distance

            shrinked = np.array(((shrinked_x1,shrinked_y1),(shrinked_x2,shrinked_y1),
                        (shrinked_x2,shrinked_y2),(shrinked_x1,shrinked_y2)))

            cv2.fillPoly(gt[:,:,0], [shrinked.astype(np.int32)], 1)

            outer_x1 = x1 - distance/2
            outer_y1 = y1 - distance/4
            outer_x2 = x2 + distance/2
            outer_y2 = y2 + distance/4

            outered = np.array(((outer_x1,outer_y1), (outer_x2,outer_y1),
                            (outer_x2,outer_y2), (outer_x1,outer_y2)))
                
            cv2.fillPoly(gt_dilate[:,:,0], [outered.astype(np.int32)], 1)
        
        if vis:

            cv2.imwrite("/home/wx/tmp_pic/detect_gt.jpg", gt*255)
            cv2.imwrite("/home/wx/tmp_pic/detect_gt_border.jpg", canvas*255)
            cv2.imwrite("/home/wx/tmp_pic/detect_border_mask.jpg", mask*255)
            cv2.imwrite("/home/wx/tmp_pic/gt_dilate.jpg",gt_dilate*255)

        target.update(detect_gt=gt,detect_gt_border=canvas, detect_border_mask=mask,
                      gt_dilate=gt_dilate)

        return image, target


class MakeBorderMap():
    shrink_ratio = 0.4 
    thresh_min = 0.3 
    thresh_max = 0.7

    def __call__(self, image, polygons, *args, **kwargs):

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        print(len(polygons))
        for i in range(len(polygons)):
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        return canvas, mask

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)

        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        
       # print(ymin_valid,ymax_valid,xmin_valid,xmax_valid)

        """
        look = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        print(look)
        """
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

       # return canvas ,mask
    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2


class Normalize(object):
    def __call__(self, image, target):
        image = F.normalize(image,[0.5,0.5,0.5],[0.5,0.5,0.5])
        return image, target


def get_transform_detect(resize_shape =(1<<11, 1<<10)):

    transforms = []
    transforms.append(GtToTensor())
    transforms.append(maskToTensor())
    transforms.append(ToTensor())
    transforms.append(Normalize())

    return Compose(transforms)

def get_prepared_transform_detect(resize_shape =(1<<11, 1<<10)):

    transforms = []
    #transforms.append(AddGtToPrepareMask())
    transforms.append(MakeSegDetectorData())
    transforms.append(Resize(out_width=resize_shape[0], 
                             out_height=resize_shape[1],
                             prepare=True)
                     )
    transforms.append(ToTensor())
    transforms.append(Normalize()) 

    return Compose(transforms)
    

def get_transforms_recognition():
    
    transforms = []
    return Compose(transforms)

if  __name__ == "__main__":

    import sys 
    sys.path.append("/home/wx/project/now_project/iocr_trainning")
    
    from datasets.invoices.invoice_ocr import End2EndInvoiceDataset,End2EndWithLoadInvoiceDataset
    from torch.utils.data import Dataset, DataLoader
    """
    dataset = End2EndInvoiceDataset(
        root='/home/wx/data/iocr_training/invoices', 
        transforms_detect=get_transform_detect(),
        #transforms_recognition=get_transforms_recognition()
    )
    print(len(dataset[0]))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    for i, (names, images, targets) in enumerate(dataloader):
        #print(i, name, "image",image,"detect_targets",detect_targets , "image_parts",image_parts,"rec_targets",rec_targets)
        print(targets)
        print(images.size())
        break

    """
    dataset = End2EndWithLoadInvoiceDataset(
        root='/home/wx/data/iocr_training/invoices', 
        transforms_detect=get_transform_detect(),
        transforms_recognition=get_transforms_recognition()
    )
    name, image, detect_target , rec_image, rec_image_target = dataset[0]
    
    print(name) 
    print(image.shape)
    print(rec_image.shape)
    print(detect_target.keys())
    print(rec_image_target['masks'].shape)

    """
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    for i, (names, images, targets) in enumerate(dataloader):
        #print(i, name, "image",image,"detect_targets",detect_targets , "image_parts",image_parts,"rec_targets",rec_targets)
        print(targets)
        print(images.size())
        break    

    """