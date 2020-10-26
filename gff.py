import numpy as np
import cv2
import argparse

R_G = 5
D_G = 5

def guidedFilter(img_i,img_p,r,eps):
    wsize = int(2*r)+1
    meanI=cv2.boxFilter(img_i,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    meanP=cv2.boxFilter(img_p,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    corrI=cv2.boxFilter(img_i*img_i,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    corrIP=cv2.boxFilter(img_i*img_p,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    varI=corrI-meanI*meanI
    covIP=corrIP-meanI*meanP
    a=covIP/(varI+eps)
    b=meanP-a*meanI
    meanA=cv2.boxFilter(a,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    meanB=cv2.boxFilter(b,ksize=(wsize,wsize),ddepth=-1,normalize=True)
    q=meanA*img_i+meanB
    return q

def GFF_GRAY(img_r, img_v):
    img_r = img_r*1./255
    img_v = img_v*1./255
    img_r_blur = cv2.blur(img_r, (31,31))
    img_v_blur = cv2.blur(img_v, (31,31))
    img_r_detail = img_r.astype(np.float) - img_r_blur.astype(np.float) 
    img_v_detail = img_v.astype(np.float) - img_v_blur.astype(np.float) 
    img_r_lap = cv2.Laplacian(img_r.astype(np.float), -1, ksize=3)
    img_v_lap = cv2.Laplacian(img_v.astype(np.float), -1, ksize=3)
    win_size = 2*R_G+1
    s1 = cv2.GaussianBlur(np.abs(img_r_lap), (win_size, win_size), R_G)
    s2 = cv2.GaussianBlur(np.abs(img_v_lap), (win_size, win_size), R_G)
    p1 = np.zeros_like(img_r)
    p2 = np.zeros_like(img_r)
    p1[s1>s2] = 1
    p2[s1<=s2] = 1
    w1_b = guidedFilter(p1, img_r.astype(np.float), 45, 0.3)
    w2_b = guidedFilter(p2, img_v.astype(np.float), 45, 0.3)
    w1_d = guidedFilter(p1, img_r.astype(np.float), 7, 0.000001)
    w2_d = guidedFilter(p2, img_v.astype(np.float), 7, 0.000001)
    w1_b_w = w1_b/(w1_b+w2_b)
    w2_b_w = w2_b/(w1_b+w2_b)
    w1_d_w = w1_d/(w1_d+w2_d)
    w2_d_w = w2_d/(w1_d+w2_d)
    fused_b = w1_b_w*img_r_blur+w2_b_w*img_v_blur
    fused_d = w1_d_w*img_r_detail+w2_d_w*img_v_detail
    img_fused = fused_b + fused_d
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(img_fused)
    
def GFF_RGB(img_r, img_v):
    fused_img = np.ones_like(img_r)
    r_R = img_r[:,:,2]
    v_R = img_v[:,:,2]
    r_G = img_r[:,:,1]
    v_G = img_v[:,:,1]
    r_B = img_r[:,:,0]
    v_B = img_v[:,:,0]
    fused_R = GFF_GRAY(r_R, v_R)
    fused_G = GFF_GRAY(r_G, v_G)
    fused_B = GFF_GRAY(r_B, v_B)
    fused_img[:,:,2] = fused_R
    fused_img[:,:,1] = fused_G
    fused_img[:,:,0] = fused_B
    return fused_img

def GFF(_rpath, _vpath):
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
            fused_img = GFF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r, img_v)
    else:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r_gray, img_v)
        else:
            fused_img = GFF_RGB(img_r, img_v)
    cv2.imshow('fused image', fused_img)
    cv2.imwrite("fused_image_gff.jpg", fused_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/IR_meting003_g.bmp' ,help='input IR image path', required=False)
    parser.add_argument('-v', type=str, default= '/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/VIS_meting003_r.bmp',help='input Visible image path', required=False)
    args = parser.parse_args()
    GFF(args.r, args.v)