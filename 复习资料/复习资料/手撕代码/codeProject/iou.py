import numpy as np

def iou(box, boxes, isMin = False): #1st框，一堆框，inMin(IOU有两种：一个除以最小值，一个除以并集)
    #计算面积：[x1,y1,x2,y3]
    box_area = (box[2] - box[0]) * (box[3] - box[1]) #原始框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  #数组代替循环

    #找交集：
    xx1 = np.maximum(box[0], boxes[:, 0]) #横坐标，左上角最大值
    yy1 = np.maximum(box[1], boxes[:, 1]) #纵坐标，左上角最大值
    xx2 = np.minimum(box[2], boxes[:, 2]) #横坐标，右下角最小值
    yy2 = np.minimum(box[3], boxes[:, 3]) #纵坐标，右小角最小值

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    #交集的面积
    inter = w * h  #对应位置元素相乘
    if isMin: #若果为False
        ovr = np.true_divide(inter, np.minimum(box_area, area)) #最小面积的IOU：O网络用
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  #并集的IOU：P和R网络用；交集/并集

    return ovr

# def iou(bbox1, bbox2, isContain = False):
#     # bbox1 = [x0,y0,x1,y1]
#     x0, y0, x1, y1 = bbox1
#     x2, y2, x3, y3 = bbox2
#
#     s1 = (x1 - x0) * (y1 - y0)
#     s2 = (x3 - x2) * (y3 - y2)
#     # 包含关系
#     if isContain:
#         return min(s1,s2)/max(s1,s2)
#     # 非包含关系
#     else:
#         w = max(0, min(x1, x3) - max(x0, x2))
#         h = max(0, min(y1, y3) - max(y0, y2))
#
#         inter = w * h
#         iou = inter / (s1 + s2 - inter)
#         return iou


if __name__ == '__main__':
    bbox1 = [1,1,3,3]
    bbox2 = [2,2,3,3]
    print(iou(bbox1,bbox2))