import numpy as np

def nms(dets, thresh):
    '''
    params: dets: bounding boxes and scores
            thres: the nms iou threshold
    return: bboxes after nms
    '''
    x1 = dets[:, 0]     # 左上角x坐标
    y1 = dets[:, 1]     # 左上角y坐标
    x2 = dets[:, 2]     # 右下角x坐标
    y2 = dets[:, 3]     # 右下角y坐标

    scores = dets[:, 4] # bboxes的置信度（scores）

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # bboxes的面积

    order = scores.argsort()[::-1]      # 根据score的大小进行降序，获得bboxes的index


    keep = []           # 要返回经过nms剩余的bboxes的index

    # Loop 
    while order.size > 0: 
        i = order[0]        # 最大score的bbox的index
        keep.append(i)      # 保存目前score最大的bbox

        # 计算其余的bbox与最大score的bbox的交际bbox的左上和右下的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集的width和height
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # intersection area
        inter_area = w * h

        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)

        # 保存IoU小于threshold的bboxes
        inds = np.where(iou <= thresh)[0]
        # 这里为什么要1呢： 因为在计算iou的时候，我们实际计算的是order[1:]和order[0]的iou，
        # 即iou的index相对与原本的index是减了1的
        order = order[inds + 1]
    return keep