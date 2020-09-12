import numpy as np
from shapely.geometry import Polygon
import pyclipper
import cv2
import torch

class MakeBorderMap():
    shrink_ratio = 0.4 
    thresh_min = 0.3 
    thresh_max = 0.7

    def __call__(self, image, polygons, *args, **kwargs):

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            try:
                self.draw_border_map(polygons[i], canvas, mask=mask)
            except Exception:
                continue
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


# 用于 生成 gt_detect 和 gt_detect_border
class MakeSegDetectorData(object):

    def __init__(self, **kwargs):
        #self.min_text_size = 8
        self.shrink_ratio = 0.4

    def __call__(self, image, targets, vis=False):
        
        quad_boxes = targets['quad_boxes']
        image_vis = image.copy()
        h, w = image.shape[0:2]
        gt = np.zeros((h, w, 1), dtype=np.float32)

        polygons_all = []

        for box in quad_boxes:
            polygons_all.append(tuple(box))

        mm = MakeBorderMap()
        canvas , mask = mm(image_vis, polygons_all)

        for i, box in enumerate(quad_boxes):

            polygon_shape = Polygon(polygons_all[i])

            distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygons_all[i]]

            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            try:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[:,:,0], [shrinked.astype(np.int32)], 1)
            except Exception:
                continue
        if vis:
            cv2.imwrite("/home/wx/tmp_pic/detect_gt.jpg", gt*255)
            cv2.imwrite("/home/wx/tmp_pic/detect_gt_border.jpg", canvas*255)
            cv2.imwrite("/home/wx/tmp_pic/detect_border_mask.jpg", mask*255)


        targets.update(image_vis=image.copy(),
                       detect_gt=torch.FloatTensor(gt).permute(2,0,1),
                       detect_gt_border=torch.FloatTensor(canvas).unsqueeze(0), 
                       detect_border_mask=torch.FloatTensor(mask).unsqueeze(0))
                      
        return image, targets


def prepocess_detect_label():

    transforms = []
    transforms.append(MakeSegDetectorData())

    return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
