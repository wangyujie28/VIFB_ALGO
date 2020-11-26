import numpy as np
import cv2
import argparse
import math

LEVEL =3
LAMBDA = 1/8
K=4

def ADF_ANISO(img):
    img_pad = np.pad(img, ((1,1), (1,1)), mode='reflect')
    img_aniso = np.zeros_like(img)
    h,w = img_pad.shape
    for i in range(1, h-1, 1):
        for j in range(1, w-1, 1):
            cn_d = img_pad[i-1,j]-img_pad[i,j]
            cs_d = img_pad[i+1,j]-img_pad[i,j]
            ce_d = img_pad[i,j+1]-img_pad[i,j]
            cw_d = img_pad[i,j-1]-img_pad[i,j]
            c_n = np.exp(-pow(cn_d/K, 2))*cn_d
            c_s = np.exp(-pow(cs_d/K, 2))*cs_d
            c_e = np.exp(-pow(ce_d/K, 2))*ce_d
            c_w = np.exp(-pow(cw_d/K, 2))*cw_d
            img_aniso[i-1,j-1] = img_pad[i,j] +LAMBDA*(c_n+c_s+c_e+c_w)
    return img_aniso


def ADF_GRAY(img_r, img_v):
    img_r  =img_r.astype(np.float)/255
    img_v =img_v.astype(np.float)/255
    img_r_base= img_r[:,:]
    img_v_base = img_v[:,:]
    for i in range(LEVEL):
        img_r_base = ADF_ANISO(img_r_base)
        img_v_base = ADF_ANISO(img_v_base)
    img_r_detail = img_r - img_r_base
    img_v_detail = img_v - img_v_base
    fused_base = (img_r_base + img_v_base) / 2
    img_r_detail_fla = img_r_detail.flatten(order='F')
    img_v_detail_fla = img_v_detail.flatten(order='F')
    img_r_mean = np.mean(img_r_detail_fla)
    img_v_mean = np.mean(img_v_detail_fla)
    img_detail_mat = np.stack((img_r_detail_fla, img_v_detail_fla), axis=-1)
    img_detail_mat = img_detail_mat - np.array((img_r_mean, img_v_mean))
    img_detail_corr = np.matmul(img_detail_mat.transpose() ,img_detail_mat)
    eig_v, eig_vec = np.linalg.eig(img_detail_corr)
    sorted_indices = np.argsort(eig_v)
    eig_vec_ch = eig_vec[:, sorted_indices[:-1-1:-1]]
    fused_detail = img_r_detail * eig_vec_ch[0][0] / (eig_vec_ch[0][0] +eig_vec_ch[1][0] )+img_v_detail * eig_vec_ch[1][0] / (eig_vec_ch[0][0] +eig_vec_ch[1][0] )
    fused_img = fused_detail + fused_base
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img


def ADF_RGB(img_r, img_v):
    r_R = img_r[:,:,2]
    r_G = img_r[:,:,1]
    r_B = img_r[:,:,0]
    v_R = img_v[:,:,2]
    v_G = img_v[:,:,1]
    v_B = img_v[:,:,0]
    fused_R= ADF_GRAY(r_R, v_R)
    fused_G= ADF_GRAY(r_G, v_G)
    fused_B= ADF_GRAY(r_B, v_B)
    fused_img = np.stack((fused_B,fused_G,fused_R), axis=-1)
    return fused_img

def ADF(r_path, v_path):
    img_r = cv2.imread(r_path)
    img_v = cv2.imread(v_path)
    if not isinstance(img_r, np.ndarray):
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray):
        print("img_v is not an image")
        return
    fused_img = None
    if len(img_r.shape)==2 or img_r.shape[-1] ==1:
        if img_r.shape[-1] ==3:
            img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
        fused_img = ADF_GRAY(img_r, img_v)
    else:
        if img_r.shape[-1] ==3:
            fused_img = ADF_RGB(img_r, img_v)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = ADF_GRAY(img_r, img_v)
    cv2.imshow("fused image", fused_img)
    cv2.imwrite("fused_image_adf.jpg", fused_img)
    cv2.waitKey(0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR",  default='/home/wang/VIFB/IV_images/IR16.png', help="path to IR image", required=False)
    parser.add_argument("--VIS",  default='/home/wang/VIFB/IV_images/VIS16.png', help="path to IR image", required=False)
    a = parser.parse_args()
    ADF(a.IR, a.VIS)