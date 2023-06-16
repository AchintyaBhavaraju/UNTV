import numpy as np
import math
import cv2 

from GuidedFilter import GuidedFilter

def getGBTransmissionESt(transmissionR,AtomsphericLightTM):
    depth_map = np.zeros(transmissionR.shape)
    for i in range(0,transmissionR.shape[0]):
        for j in range(0, transmissionR.shape[1]):
            depth_map[i,j]  = math.log(transmissionR[i,j],0.82)

    transmissionG = 0.93 ** depth_map
    transmissionB = 0.95 ** depth_map

    return transmissionB,transmissionG,depth_map


def getMinChannel(img,AtomsphericLight):
    imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]), dtype=np.float16)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 1
            for k in range(0, 3):
                imgNormalization = img.item((i, j, k)) / AtomsphericLight[k]
                if imgNormalization < localMin:
                    localMin = imgNormalization
            imgGrayNormalization[i, j] = localMin
    return imgGrayNormalization


def getTransmission(img,AtomsphericLight ,blockSize): #4th parameter color
    img = np.float16(img)
    #Calculate the Minumum Channel
    img = getMinChannel(img,AtomsphericLight)
    #Color = tranmissionG + transmssionB
    AtomsphericLight = AtomsphericLight / 255.0
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 1
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    imgDark = np.zeros((img.shape[0], img.shape[1]))
    localMin = 1
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin
        transmission = (1 - imgDark) / (1 - 0.1 / np.max(AtomsphericLight))
    transmission = np.clip(transmission, 0.1, 0.9)

    return transmission


def  Refinedtransmission(transmissionB,transmissionG,transmissionR_Stretched,img):
    gimfiltR = 50
    eps = 10 ** -3

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmissionR_Stretched = guided_filter.filter(transmissionR_Stretched)
    transmissionG = guided_filter.filter(transmissionG)
    transmissionB = guided_filter.filter(transmissionB)

    transmission = np.zeros(img.shape)
    transmission[:, :, 0] = transmissionB
    transmission[:, :, 1] = transmissionG
    transmission[:, :, 2] = transmissionR_Stretched
    transmission = np.clip(transmission,0.05, 0.95)

    return transmission
