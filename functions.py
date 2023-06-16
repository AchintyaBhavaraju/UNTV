import os
import numpy as np
import datetime
import cv2
import math
import natsort

def Sat_max(img):
    height = len(img)
    width = len(img[0])
    # print('img[0,0,:]',img[0,0,:])
    Sat = np.zeros((height,width ))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(np.max(img[i,j,:]) == 0):
                Sat[i,j] = 1
            else:
                Sat[i, j] = (np.max(img[i,j,:]) - np.min(img[i,j,:]))/np.max(img[i,j,:])
    Sat = 1 - Sat

    # lamba = 1 - np.mean(Sat)
    lamba = 1

    Sat = Sat * lamba
    return Sat


def color_correction(r,u_r,u_ref,L2):
    L1 = np.max(r)
    gainFactor = L1 * (u_r/ u_ref) +L2
    Out = r / gainFactor
    return Out

def OptimalParameter(sceneRadiance):
    img = np.float64(sceneRadiance / 255)
    b, g, r = cv2.split(img)

    u_r = np.sum(r)
    u_g = np.sum(g)
    u_b = np.sum(b)
    u_ref = (u_r ** 2 + u_g ** 2 + u_b ** 2) ** 0.5
    L2 = 0.25
    r = color_correction(r, u_r, u_ref, L2)
    g = color_correction(g, u_g, u_ref, L2)
    b = color_correction(b, u_b, u_ref, L2)

    sceneRadiance = np.zeros((img.shape), 'float64')
    sceneRadiance[:, :, 0] = b
    sceneRadiance[:, :, 1] = g
    sceneRadiance[:, :, 2] = r
    sceneRadiance = sceneRadiance * 255
    sceneRadiance = np.clip(sceneRadiance,0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance

def depthMap(img):

    theta_0 = 0.51157954
    theta_1 = 0.50516165
    theta_2 = -0.90511117
    img = img / 255.0
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    x_1 = np.maximum(g, b)
    x_2 = r

    Deptmap = theta_0 + theta_1 * x_1 + theta_2 * x_2
    return Deptmap


def  minDepth(img, BL):
    img = img/255.0
    BL = BL/255.0
    Max = []
    img = np.float32(img)
    for i in range(0,3):
        Max_Abs =  np.absolute(img[i] - BL[i])
        Max_I = np.max(Max_Abs)
        Max_B = np.max([BL[i],(1 -BL[i])])
        temp  = Max_I / Max_B
        Max.append(temp)
    K_b = np.max(Max)
    min_depth = 1 - K_b

    return min_depth

def Depth_TM(img, AtomsphericLight):

    DepthMap = depthMap(img)
    t0, t1 = 0.05, 0.95
    DepthMap = DepthMap.clip(t0, t1)
    d_0 = minDepth(img, AtomsphericLight)

    d_f = 8 * (DepthMap + d_0)
    TM_R_modified = 0.85 ** d_f
    return TM_R_modified


def sceneRadianceRGB(img, transmission, AtomsphericLight):
    sceneRadiance = np.zeros(img.shape)
    img = np.float32(img)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission[:, :, i]  + AtomsphericLight[i]


    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance














