import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint #多边形


def calc_iou(quad1, quad2):
    
    poly1 = Polygon(quad1).convex_hull #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    poly2 = Polygon(quad2).convex_hull
    union_poly = np.concatenate((quad1,quad2))  #合并两个box坐标，变为8*2

    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  #相交面积
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
                #iou = float(inter_area) / (union_area-inter_area) #错了
                iou=float(inter_area) / union_area
                # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
                # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积 
                # 第二种： 交集 / 并集（常见矩形框IOU计算方式） 
        except shapely.geos.TopologicalError:
            iou = 0
    
    return iou

