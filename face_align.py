import face_alignment
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)


def face_align1(face_line):
    preds = fa.get_landmarks(face_line)
    # face_gray1 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # face_gray2 = face_gray1
    # m, n = face_gray1.shape
    # face_div = np.zeros((m, n))
    x = preds[0][:,0]
    y = preds[0][:,1]

    brow_lx = x[17:22]
    brow_ly = y[17:22]
    brow_rx = x[22:27]
    brow_ry = y[22:27]
    brow_lind1 = brow_ly.argmin()
    brow_rind1 = brow_ry.argmin()
    brow_lind2 = brow_lx.argmin()
    brow_rind2 = brow_rx.argmax()
    brow_lx1, brow_ly1 = brow_lx[brow_lind1], brow_ly[brow_lind1]
    brow_rx1, brow_ry1 = brow_rx[brow_rind1], brow_ry[brow_rind1]
    brow_lx2, brow_ly2 = brow_lx[brow_lind2], brow_ly[brow_lind2]
    brow_rx2, brow_ry2 = brow_rx[brow_rind2], brow_ry[brow_rind2]

    # face_div[int(brow_ly1):m, :] += 1
    # face_div[int(brow_ry1):m, :] += 1
    # face_div[:, int(brow_lx2):n] += 1
    # face_div[:, 0:int(brow_rx2)] += 1

    eye_lx = x[36:42]
    eye_ly = y[36:42]
    eye_rx = x[42:48]
    eye_ry = y[42:48]
    eye_lind = eye_lx.argmin()
    eye_rind = eye_rx.argmax()
    eye_lx1, eye_ly1 = eye_lx[eye_lind], eye_ly[eye_lind]
    eye_rx1, eye_ry1 = eye_rx[eye_rind], eye_ry[eye_rind]

    # face_div[:, int(eye_lx1):n] += 1
    # face_div[:, 0:int(eye_rx1)] += 1

    nose_downx = x[31:36]
    nose_downy = y[31:36]
    nose_lind = nose_downx.argmin()
    nose_rind = nose_downx.argmax()
    nose_lx1, nose_ly1 = nose_downx[nose_lind], nose_downy[nose_lind]
    nose_rx1, nose_ry1 = nose_downx[nose_rind], nose_downy[nose_rind]

    # face_div[:, int(nose_lx1):n] += 1
    # face_div[:, 0:int(nose_rx1)] += 1

    mouse_x = x[48:60]
    mouse_y = y[48:60]
    mouse_lind = mouse_x.argmin()
    mouse_rind = mouse_x.argmax()
    mouse_ind = mouse_y.argmax()
    mouse_lx1, mouse_ly1 = mouse_x[mouse_lind], mouse_y[mouse_lind]
    mouse_rx1, mouse_ry1 = mouse_x[mouse_rind], mouse_y[mouse_rind]
    mouse_dx1, mouse_dy1 = mouse_x[mouse_ind ], mouse_y[mouse_ind ]

    # face_div[:, int(mouse_lx1):n] += 1
    # face_div[:, 0:int(mouse_rx1)] += 1
    # face_div[0:int(mouse_dy1), :] += 1



    # x_point = [brow_lx1, brow_rx1, brow_rx2, eye_rx1, nose_rx1, mouse_rx1, mouse_dx1, mouse_lx1, nose_lx1, eye_lx1, brow_lx2, brow_lx1]
    # y_point = [brow_ly1, brow_ry1, brow_ry2, eye_ry1, nose_ry1, mouse_ry1, mouse_dy1, mouse_ly1, nose_ly1, eye_ly1, brow_ly2, brow_ly1]

    point_1 = (brow_lx1, brow_ly1)
    point_2 = (brow_rx1, brow_ry1)
    point_3 = (brow_rx2, brow_ry2)
    point_4 = (eye_rx1, eye_ry1)
    point_5 = (nose_rx1, nose_ry1)
    point_6 = (mouse_rx1, mouse_ry1)
    point_7 = (mouse_dx1, mouse_dy1)
    point_8 = (mouse_lx1, mouse_ly1)
    point_9 = (nose_lx1, nose_ly1)
    point_10 = (eye_lx1, eye_ly1)
    point_11 = (brow_lx2, brow_ly2)

    cv2.line(face_line, point_1, point_2, (255,0,0))
    cv2.line(face_line, point_2, point_3, (255,0,0))
    cv2.line(face_line, point_3, point_4, (255,0,0))
    cv2.line(face_line, point_4, point_5, (255,0,0))
    cv2.line(face_line, point_5, point_6, (255,0,0))
    cv2.line(face_line, point_6, point_7, (255,0,0))
    cv2.line(face_line, point_7, point_8, (255,0,0))
    cv2.line(face_line, point_8, point_9, (255,0,0))
    cv2.line(face_line, point_9, point_10, (255,0,0))
    cv2.line(face_line, point_10, point_11, (255,0,0))
    cv2.line(face_line, point_11, point_1, (255,0,0))

    return face_line, brow_lx2, brow_ly2, brow_rx2, mouse_dy1
    # face_div = (face_div-9) * 255
    # cv2.imshow('face', face_gray1)
    # cv2.imshow('face_1', face_div)
    # cv2.waitKey()

# plt.figure()
# plt.axis('off')
# plt.imshow(face_gray)
# plt.plot(x_point, y_point, 'w-')
# plt.savefig("./test/assets/03.jpg")
# plt.show()
