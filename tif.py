import numpy as np
import cv2
import argparse

def TIF_GRAY(img_r, img_v):
    img_r_blur = cv2.blur(img_r, (35,35))
    img_v_blur = cv2.blur(img_v, (35,35))
    img_r_median = cv2.medianBlur(img_r, 3)
    img_v_median = cv2.medianBlur(img_v, 3)
    img_r_detail = img_r*1. - img_r_blur*1.
    img_v_detail = img_v*1. - img_v_blur*1.
    img_r_the = cv2.pow(cv2.absdiff(img_r_median,img_r_blur), 2)
    img_v_the = cv2.pow(cv2.absdiff(img_v_median ,img_v_blur),2)
    img_r_weight = cv2.divide(img_r_the*1.,img_r_the*1.+img_v_the*1.+0.000001)
    img_v_weight = 1- img_r_weight
    img_base_fused = (img_r_blur*1.  + img_v_blur*1.) / 2
    img_detail_fused = img_r_weight * img_r_detail + img_v_weight * img_v_detail
    img_fused_tmp = (img_base_fused  + img_detail_fused).astype(np.int32)
    #first method to change <0 to 0 and > 255 to 255
    img_fused_tmp[img_fused_tmp<0] = 0
    img_fused_tmp[img_fused_tmp>255]=255
    #second method to change value to[0,255] using minmax method
    #cv2.normalize(img_fused_tmp,img_fused_tmp,0,255,cv2.NORM_MINMAX)
    img_fused = cv2.convertScaleAbs(img_fused_tmp)
    return img_fused
    

def TIF_RGB(img_r, img_v):
    fused_img = np.ones_like(img_r)
    r_R = img_r[:,:,2]
    v_R = img_v[:,:,2]
    r_G = img_r[:,:,1]
    v_G = img_v[:,:,1]
    r_B = img_r[:,:,0]
    v_B = img_v[:,:,0]
    fused_R = TIF_GRAY(r_R, v_R)
    fused_G = TIF_GRAY(r_G, v_G)
    fused_B = TIF_GRAY(r_B, v_B)
    fused_img[:,:,2] = fused_R
    fused_img[:,:,1] = fused_G
    fused_img[:,:,0] = fused_B
    return fused_img



def TIF(_rpath, _vpath):
    img_r = cv2.imread(_rpath)
    img_v = cv2.imread(_vpath)
    if not isinstance(img_r, np.ndarray) :
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray) :
        print('img_v is null')
        return
    if img_r.shape[0] != img_v.shape[0]  or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        return
    fused_img = None
    if len(img_r.shape)  < 3 or img_r.shape[2] ==1:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            fused_img = TIF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = TIF_GRAY(img_r, img_v)
    else:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = TIF_GRAY(img_r_gray, img_v)
        else:
            fused_img = TIF_RGB(img_r, img_v)
    cv2.imshow('fused image', fused_img)
    cv2.imwrite("fused_image_tif.jpg", fused_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/IR_meting003_g.bmp' ,help='input IR image path', required=False)
    parser.add_argument('-v', type=str, default= '/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/VIS_meting003_r.bmp',help='input Visible image path', required=False)
    args = parser.parse_args()
    TIF(args.r, args.v)