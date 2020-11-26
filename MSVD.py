import numpy as np
import cv2
import argparse
import math

LEVEL =1

def MSVD_SVD(img, h, w):
    '''
    tl = img[0:h//2, 0:w//2]             #top left
    tr = img[0:h//2, w//2:]             #top right
    bl = img[h//2:, 0:w//2]             #bottom left
    br = img[h//2:, w//2:]            #bottom right
    tl_r = tl.flatten(order="F")
    tr_r = tr.flatten(order="F")
    bl_r = bl.flatten(order="F")
    br_r = br.flatten(order="F")
    t_x = np.stack((tl_r, bl_r, tr_r , br_r), axis=0)
    '''
    nh = h //2
    nw = w //2
    t_x = np.zeros((4, nh*nw))
    for j in range(nw):
        for i in range(nh):
            t_x[:, j*nh+i] = img[2*i:2*(i+1), 2*j:2*(j+1)].flatten(order="F")
    x1 = np.matmul(t_x,t_x.transpose())
    eig_v, eig_vec = np.linalg.eig(x1)
    sorted_indices = np.argsort(eig_v)
    eig_vec_ch = eig_vec[:, sorted_indices[:-4-1:-1]]
    eig_vec_ch_t = eig_vec_ch.transpose()
    t_d = np.matmul(eig_vec_ch_t, t_x)
    return t_d[0], t_d[1], t_d[2], t_d[3], eig_vec_ch


def MSVD_GRAY(img_r, img_v):
    img_r  =img_r.astype(np.float)/255
    img_v =img_v.astype(np.float)/255
    h, w = img_r.shape
    nh = (h // int(pow(2, LEVEL)) )* int(pow(2, LEVEL))
    nw = (w // int(pow(2, LEVEL)) )* int(pow(2, LEVEL))
    h, w = nh, nw
    img_r = cv2.resize(img_r, (w,h))
    img_v = cv2.resize(img_v, (w,h))
    fer_rl = None
    fer_vl = None
    theta_rl = []
    theta_vl = []
    u_rl = []
    u_vl = []
    img_r_copy = img_r[:,:]
    img_v_copy = img_v[:,:]
    for i in range(LEVEL):
        fei_r, theta_r_1, theta_r_2, theta_r_3, u_r = MSVD_SVD(img_r_copy, h, w)
        fei_v, theta_v_1, theta_v_2, theta_v_3, u_v = MSVD_SVD(img_v_copy,  h, w)
        h = h // 2
        w = w // 2
        img_r_copy =fei_r.reshape((w,h)).transpose()
        img_v_copy = fei_v.reshape((w,h)).transpose()
        fer_rl = fei_r[:]
        fer_vl = fei_v[:]
        theta_rl.append((theta_r_1, theta_r_2, theta_r_3))
        theta_vl.append((theta_v_1, theta_v_2, theta_v_3))
        u_rl.append(u_r)
        u_vl.append(u_v)
    fused_fei = None
    for i in range(LEVEL-1,-1,-1):
        if i == LEVEL-1:
            fused_fei = (fer_rl+fer_vl)/ 2
        theta_0_effi = (np.abs(theta_vl[i][0])>np.abs(theta_rl[i][0])).astype(np.float)
        theta_1_effi = (np.abs(theta_vl[i][1])>np.abs(theta_rl[i][1])).astype(np.float)
        theta_2_effi = (np.abs(theta_vl[i][2])>np.abs(theta_rl[i][2])).astype(np.float)
        fused_theta_0=theta_0_effi*theta_vl[i][0]+(1-theta_0_effi)*theta_rl[i][0]
        fused_theta_1=theta_1_effi*theta_vl[i][1]+(1-theta_1_effi)*theta_rl[i][1]
        fused_theta_2=theta_2_effi*theta_vl[i][2]+(1-theta_2_effi)*theta_rl[i][2]
        fused_u = (u_rl[i]+u_vl[i]) / 2
        fused_fei = np.stack((fused_fei, fused_theta_0, fused_theta_1, fused_theta_2), axis = 0)
        fused_fei = np.matmul(fused_u, fused_fei)
        th = h
        tw = w
        h *= 2
        w *= 2
        fused_img = np.zeros([h,w])
        for k in range(tw):
            for l in range(th):
                fused_img[l*2, k*2] = fused_fei[0][l+k*th]
                fused_img[l*2+1, k*2] = fused_fei[1][l+k*th]
                fused_img[l*2,k*2+1] = fused_fei[2][l+k*th]
                fused_img[l*2+1, k*2+1] = fused_fei[3][l+k*th]
        fused_fei = fused_img.flatten(order="F")
        '''
        if i == LEVEL-1:
            fused_fei = fer_vl
        fused_theta_0=theta_vl[i][0]
        fused_theta_1=theta_vl[i][1]
        fused_theta_2=theta_vl[i][2]
        fused_u = u_vl[i]
        fused_fei = np.stack((fused_fei, fused_theta_0, fused_theta_1, fused_theta_2), axis = 0)
        fused_fei = np.matmul(fused_u, fused_fei)
        th = h
        tw = w
        h *= 2
        w *= 2
        fused_img = np.zeros([h,w])
        for k in range(tw):
            for l in range(th):
                fused_img[l*2, k*2] = fused_fei[0][l+k*th]
                fused_img[l*2+1, k*2] = fused_fei[1][l+k*th]
                fused_img[l*2,k*2+1] = fused_fei[2][l+k*th]
                fused_img[l*2+1, k*2+1] = fused_fei[3][l+k*th]
        fused_fei = fused_img.flatten(order="F")
    '''
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img


def MSVD_RGB(img_r, img_v):
    r_R = img_r[:,:,2]
    r_G = img_r[:,:,1]
    r_B = img_r[:,:,0]
    v_R = img_v[:,:,2]
    v_G = img_v[:,:,1]
    v_B = img_v[:,:,0]
    fused_R= MSVD_GRAY(r_R, v_R)
    fused_G= MSVD_GRAY(r_G, v_G)
    fused_B= MSVD_GRAY(r_B, v_B)
    fused_img = np.stack((fused_B,fused_G,fused_R), axis=-1)
    '''
    r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
    fused_img = MSVD_GRAY(r_gray, v_gray)
    '''
    return fused_img

def MSVD(r_path, v_path):
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
        fused_img = MSVD_GRAY(img_r, img_v)
    else:
        if img_r.shape[-1] ==3:
            fused_img = MSVD_RGB(img_r, img_v)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = MSVD_GRAY(img_r, img_v)
    cv2.imshow("fused image", fused_img)
    cv2.imwrite("fused_image_msvd.jpg", fused_img)
    cv2.waitKey(0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR",  default='/home/wang/VIFB/IV_images/IR1.png', help="path to IR image", required=False)
    parser.add_argument("--VIS",  default='/home/wang/VIFB/IV_images/VIS1.png', help="path to IR image", required=False)
    a = parser.parse_args()
    MSVD(a.IR, a.VIS)