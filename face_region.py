import cv2
import numpy as np
from face_align import face_align1

def max0(data):
    if data > 0:
        return data
    else:
        return 0


def face_region1(face_line, brow_lx2, brow_ly2, brow_rx2, mouse_dy1):
    img = face_line
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    x, y = max0(int(brow_lx2-15)), max0(int(brow_ly2-15))
    w = int(brow_rx2-brow_lx2+30)
    h = int(mouse_dy1-brow_ly2+30)
    cv2.imshow('img', img)
    cv2.waitKey()

    rect = (x, y, w, h)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]

    return img


# cv2.imshow('img', img)
# cv2.waitKey()