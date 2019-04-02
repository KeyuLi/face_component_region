import face_alignment
import numpy as np
import os
import cv2
from skimage import exposure
import dlib



detector = dlib.get_frontal_face_detector()
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
img_size = 178
input_url = "./faceData/"
output_url = "./nose_region/"

def int0(data):
    m, n = data.shape

    for x in range(m):
        for y in range(n):
            data[x, y] = int(data[x, y])
    return data

def max0(data):
    if data > 0:
        return data
    else:
        return 0

def max1(data):
    if data < img_size:
        return data
    else:
        return img_size - 1

def pix_light(origin, region, ingamma, degamma):
    m,n,z = origin.shape
    origin_out = np.zeros((m,n,z))
    for x in range(m):
        for y in range(n):
            if region[x,y,:].all() == 0:
                origin_out[x,y,:] = ((origin[x,y,:]/255)**degamma)*255
            else:
                origin_out[x,y,:] = ((origin[x,y,:]/255)**ingamma)*255
    dtype = origin.dtype.type
    origin_out = dtype(origin_out)
    return origin_out

if __name__ == '__main__':
    lists = os.listdir(input_url)
    lists.sort()
    face_img = np.zeros((img_size, img_size))
    face_line = np.zeros((img_size, img_size))
    face_region = np.zeros((img_size, img_size))
    face_light = np.zeros((img_size, img_size))

    for i in range(0, len(lists)):
        data_url = input_url + lists[i]
        img = cv2.imread(data_url)
        # faces = detector(img, 1)
        # if len(faces) != 1:
        #
        #     continue

        # face = max(faces, key=lambda rect: rect.width() * rect.height())
        # [x1, x2, y1, y2] = [face.left(), face.right(), face.top(), face.bottom()]
        # img1 = img[max0(y1): y2, max0(x1):  x2]
        # face_img = cv2.resize(img1, (img_size, img_size))
        # face_line = face_img


# face_alignment

        preds = fa.get_landmarks(img)
        # print(preds)
        if preds == None:
            cv2.imwrite(output_url + lists[i], img)
            print('ok', i)
            continue
        face_line = exposure.adjust_gamma(face_line, 1.0)

        x = preds[0][:, 0]
        y = preds[0][:, 1]

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


        eye_lx = x[36:42]
        eye_ly = y[36:42]
        eye_rx = x[42:48]
        eye_ry = y[42:48]
        eye_lind = eye_lx.argmin()
        eye_rind = eye_rx.argmax()
        # eye_dlind = eye_ly.argmax()
        # eye_drind = eye_ry.argmax()

        eye_lx1, eye_ly1 = eye_lx[eye_lind], eye_ly[eye_lind]
        eye_rx1, eye_ry1 = eye_rx[eye_rind], eye_ry[eye_rind]
        # eye_dlx1, eye_dly1 = eye_lx[eye_dlind], eye_ly[eye_dlind]
        # eye_drx1, eye_dry1 = eye_rx[eye_drind], eye_ry[eye_drind]



        nose_downx = x[31:36]
        nose_downy = y[31:36]
        nose_x = x[27:31]
        nose_y = y[27:31]
        nose_lind = nose_downx.argmin()
        nose_rind = nose_downx.argmax()
        nose_yind = nose_y.argmax()
        nose_lx1, nose_ly1 = nose_downx[nose_lind], nose_downy[nose_lind]
        nose_rx1, nose_ry1 = nose_downx[nose_rind], nose_downy[nose_rind]
        nose_x1, nose_y1 = nose_x[nose_yind], nose_y[nose_yind]



        mouse_x = x[48:60]
        mouse_y = y[48:60]
        mouse_lind = mouse_x.argmin()
        mouse_rind = mouse_x.argmax()
        mouse_ind = mouse_y.argmin()
        mouse_lx1, mouse_ly1 = mouse_x[mouse_lind], mouse_y[mouse_lind]
        mouse_rx1, mouse_ry1 = mouse_x[mouse_rind], mouse_y[mouse_rind]
        mouse_dx1, mouse_dy1 = mouse_x[mouse_ind], mouse_y[mouse_ind]








        # cv2.line(face_line, point_1, point_2, (255, 255, 255), 2)
        # cv2.line(face_line, point_2, point_3, (255, 255, 255), 2)
        # cv2.line(face_line, point_3, point_4, (255, 255, 255), 2)
        cv2.line(face_line, (brow_lx[0], brow_ly[0]), (brow_lx[1], brow_ly[1]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_lx[1], brow_ly[1]), (brow_lx[2], brow_ly[2]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_lx[2], brow_ly[2]), (brow_lx[3], brow_ly[3]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_lx[3], brow_ly[3]), (brow_lx[4], brow_ly[4]), (255, 255, 255), 2)

        cv2.line(face_line, (brow_lx[4], brow_ly[4]), (brow_rx[0], brow_ry[0]), (255, 255, 255), 2)

        cv2.line(face_line, (brow_rx[0], brow_ry[0]), (brow_rx[1], brow_ry[1]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_rx[1], brow_ry[1]), (brow_rx[2], brow_ry[2]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_rx[2], brow_ry[2]), (brow_rx[3], brow_ry[3]), (255, 255, 255), 2)
        cv2.line(face_line, (brow_rx[3], brow_ry[3]), (brow_rx[4], brow_ry[4]), (255, 255, 255), 2)

        cv2.line(face_line, (brow_rx[4], brow_ry[4]), (eye_rx[3], eye_ry[3]), (255, 255, 255), 2)

        cv2.line(face_line, (eye_rx[3], eye_ry[3]), (eye_rx[4], eye_ry[4]), (255, 255, 255), 2)
        cv2.line(face_line, (eye_rx[4], eye_ry[4]), (eye_rx[5], eye_ry[5]), (255, 255, 255), 2)
        cv2.line(face_line, (eye_rx[5], eye_ry[5]), (eye_rx[0], eye_ry[0]), (255, 255, 255), 2)

        cv2.line(face_line, (eye_rx[0], eye_ry[0]), (nose_downx[4], nose_downy[4]), (255, 255, 255), 2)

        cv2.line(face_line, (nose_downx[4], nose_downy[4]), (mouse_x[6], mouse_y[6]), (255, 255, 255), 2)

        #
        # #############eye################
        # cv2.line(face_line, point_4, point_12, (255, 255, 255), 2)
        # cv2.line(face_line, point_12, point_10, (255, 255, 255), 2)
        # cv2.line(face_line, point_10, point_11, (255, 255, 255), 2)
        # cv2.line(face_line, point_11, point_1, (255, 255, 255), 2)
        ###############

        ###################mouse######################
        # cv2.line(face_line, point_12, point_6, (255, 255, 255), 2)
        # cv2.line(face_line, point_6, point_7, (255, 255, 255), 2)
        # cv2.line(face_line, point_7, point_8, (255, 255, 255), 2)
        # cv2.line(face_line, point_8, point_12, (255, 255, 255), 2)

        ###############################mouse#############
        # cv2.line(face_line, (mx1, my1), (mx2, my2), (255,255,255), 2)
        # cv2.line(face_line, (mx2, my2), (mx3, my3), (255,255,255), 2)
        # cv2.line(face_line, (mx3, my3), (mx4, my4), (255,255,255), 2)
        # cv2.line(face_line, (mx4, my4), (mx5, my5), (255,255,255), 2)
        # cv2.line(face_line, (mx5, my5), (mx6, my6), (255,255,255), 2)
        # cv2.line(face_line, (mx6, my6), (mx7, my7), (255,255,255), 2)
        # cv2.line(face_line, (mx7, my7), (mx8, my8), (255,255,255), 2)
        # cv2.line(face_line, (mx8, my8), (mx9, my9), (255,255,255), 2)
        # cv2.line(face_line, (mx9, my9), (mx10, my10), (255,255,255), 2)
        # cv2.line(face_line, (mx10, my10), (mx11, my11), (255,255,255), 2)
        # cv2.line(face_line, (mx11, my11), (mx12, my12), (255,255,255), 2)
        # cv2.line(face_line, (mx12, my12), (mx1, my1), (255,255,255), 2)
        #################region####################

        # cv2.line(face_line, point_4, point_5, (255, 255, 255), 2)
        # cv2.line(face_line, point_5, point_6, (255, 255, 255), 2)
        # cv2.line(face_line, point_6, point_7, (255, 255, 255), 2)
        # cv2.line(face_line, point_7, point_8, (255, 255, 255), 2)
        # cv2.line(face_line, point_8, point_9, (255, 255, 255), 2)
        # cv2.line(face_line, point_9, point_10, (255, 255, 255), 2)
        # cv2.line(face_line, point_10, point_11, (255, 255, 255), 2)
        # cv2.line(face_line, point_11, point_1, (255, 255, 255), 2)
        ################################################


        cv2.imshow('img', face_line)
        cv2.waitKey()

        # face_region


        w2 = max1(int(mouse_rx1 - mouse_lx1 + 32))
        h2 = max1(int(mouse_dy1 - eye_ly[3] + 32))

        if w2 <= 0 or h2 <= 0:
            cv2.imwrite(output_url + lists[i], img)
            print(i)
            continue
        if i % 1000 ==0:
            print(i)