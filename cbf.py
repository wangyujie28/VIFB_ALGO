import numpy as np
import cv2
import argparse
import math

cov_wsize = 5
sigmas = 1.8
sigmar = 25
ksize = 11

def gaussian_kernel_2d_opencv(kernel_size = 11,sigma = 1.8):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky)) 

def bilateralFilterEx(img_r, img_v):
    #edge solved
    win_size = ksize//2
    img_r_copy = None
    img_v_copy = None
    img_r_copy = cv2.copyTo(img_r, None)
    img_v_copy = cv2.copyTo(img_v, None)
    img_r_cbf = np.ones_like(img_r, dtype=np.float)
    img_v_cbf = np.ones_like(img_r, dtype=np.float)
    img_r_copy = np.pad(img_r_copy, (win_size, win_size), 'reflect')
    img_v_copy = np.pad(img_v_copy, (win_size, win_size), 'reflect')
    gk = gaussian_kernel_2d_opencv()
    for i in range(win_size, win_size+img_r.shape[0]):
        for j in range(win_size, win_size+img_r.shape[1]):
            sumr1 = 0.
            sumr2 = 0.
            sumv1 = 0.
            sumv2 = 0.
            img_r_cdis = img_r_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0- img_r_copy[i,j]*1.0
            img_v_cdis = img_v_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0- img_v_copy[i,j]*1.0
            sumr1 = np.sum(np.exp(-img_v_cdis*img_v_cdis) *gk/ (2*sigmar*sigmar) )
            sumv1 = np.sum(np.exp(-img_r_cdis*img_r_cdis) *gk/ (2*sigmar*sigmar) )
            sumr2 = np.sum(np.exp(-img_v_cdis*img_v_cdis) *gk*img_r_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0/ (2*sigmar*sigmar) )
            sumv2 = np.sum(np.exp(-img_r_cdis*img_r_cdis) *gk*img_v_copy[i-win_size:i+win_size+1, j-win_size:j+win_size+1] *1.0/ (2*sigmar*sigmar) )
            img_r_cbf[i-win_size,j-win_size] = sumr2 / sumr1
            img_v_cbf[i-win_size,j-win_size] = sumv2 / sumv1
    return (img_r*1. - img_r_cbf, img_v*1. - img_v_cbf)

def CBF_WEIGHTS(img_r_d, img_v_d):
    win_size = cov_wsize // 2
    img_r_weights = np.ones_like(img_r_d, dtype=np.float)
    img_v_weights= np.ones_like(img_v_d, dtype=np.float)
    img_r_d_pad = np.pad(img_r_d, (win_size, win_size), 'reflect')
    img_v_d_pad = np.pad(img_v_d, (win_size, win_size), 'reflect')
    for i in range(win_size, win_size+img_r_d.shape[0]):
        for j in range(win_size, win_size+img_r_d.shape[1]):
            npt_r = img_r_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            npt_v = img_v_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            npt_r_V = npt_r - np.mean(npt_r, axis=0)
            npt_r_V = npt_r_V*npt_r_V.transpose()
            npt_r_H = npt_r.transpose() - np.mean(npt_r, axis=1)
            npt_r_H = npt_r_H*npt_r_H.transpose()
            npt_v_V = npt_v - np.mean(npt_v, axis=0)
            npt_v_V = npt_v_V*npt_v_V.transpose()
            npt_v_H = npt_v.transpose() - np.mean(npt_v, axis=1)
            npt_v_H = npt_v_H*npt_v_H.transpose()
            img_r_weights[i-win_size,j-win_size] = np.trace(npt_r_H) + np.trace(npt_r_V) 
            img_v_weights[i-win_size,j-win_size] = np.trace(npt_v_H) + np.trace(npt_v_V) 
    return img_r_weights, img_v_weights

def CBF_GRAY(img_r, img_v):
    img_r_d, img_v_d = bilateralFilterEx(img_r, img_v)
    img_r_weights, img_v_weights = CBF_WEIGHTS(img_r_d, img_v_d)
    img_fused =(img_r*1. * img_r_weights + img_v*1.*img_v_weights) /(img_r_weights+img_v_weights)
    img_fused = cv2.convertScaleAbs(img_fused)
    return img_fused

def CBF_RGB(img_r, img_v):
    img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
    return CBF_GRAY(img_r_gray, img_v_gray)

def CBF(_rpath, _vpath):
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
            fused_img = CBF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = CBF_GRAY(img_r, img_v)
    else:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = CBF_GRAY(img_r_gray, img_v)
        else:
            fused_img = CBF_RGB(img_r, img_v)
    cv2.imshow('fused image', fused_img)
    cv2.imwrite("fused_image.jpg", fused_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/IR_meting003_g.bmp' ,help='input IR image path', required=False)
    parser.add_argument('-v', type=str, default= '/home/wang/VIFB/TNO_Image_Fusion_Dataset/Athena_images/2_men_in_front_of_house/VIS_meting003_r.bmp',help='input Visible image path', required=False)
    args = parser.parse_args()
    CBF(args.r, args.v)