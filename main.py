import os
import datetime
import numpy as np
import cv2
import natsort

from transmission import *
from functions import *

from global_histogram_stretching import stretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

starttime = datetime.datetime.now()
path = ("/Users/achintya/Desktop/UNTV/top")
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread("top/01009.jpg")
        blockSize = 9
        height = len(img)
        width = len(img[0])
        gimfiltR = 50  # Radius size of guided filter
        eps = 10 ** -3  # Epsilon value of guided filter
        Nrer = [0.95, 0.93, 0.85] # Normalized residual energy ratio of G-B-R channels

        AtomsphericLight = np.zeros(3)
        AtomsphericLight[0] = (1.13 * np.mean(img[:, :, 0])) + 1.11 * np.std(img[:, :, 0]) - 25.6
        AtomsphericLight[1] = (1.13 * np.mean(img[:, :, 1])) + 1.11 * np.std(img[:, :, 1]) - 25.6
        AtomsphericLight[2] = 140 / (1 + 14.4 * np.exp(-0.034 * np.median(img[:, :, 2])))
        AtomsphericLight = np.clip(AtomsphericLight, 5, 250)
        transmissionR = getTransmission(img, AtomsphericLight, blockSize)
        TM_R_modified = Depth_TM(img, AtomsphericLight)
        TM_R_modified_Art = Sat_max(img)
        transmissionR_new = np.copy(transmissionR)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if(transmissionR_new[i, j] > TM_R_modified[i, j]):
                    transmissionR_new[i, j] = TM_R_modified[i, j]
                if(transmissionR_new[i, j] < TM_R_modified_Art[i, j]):
                    transmissionR_new[i, j] = TM_R_modified_Art[i, j]

        transmissionR_Stretched = stretching(transmissionR_new, height, width)
        transmissionB, transmissionG, depth_map = getGBTransmissionESt(transmissionR_Stretched, AtomsphericLight)
        transmission = Refinedtransmission(transmissionB, transmissionG, transmissionR_Stretched, img)
        #give transmission map of both
        transmissionB_image = (transmissionB * 255).astype(np.uint8)
        transmissionG_image = (transmissionG * 255).astype(np.uint8)

        # Mark the transmission maps with a red dot
        # red_dot = (0, 0, 255)  # Red color for the dot
        # cv2.circle(transmissionB_image, (10, 10), 3, red_dot, -1)
        # cv2.circle(transmissionG_image, (10, 10), 3, red_dot, -1)

        # Display the transmission maps
        cv2.imshow("Transmission B", transmissionB_image)
        cv2.imshow("Transmission G", transmissionG_image)
        cv2.waitKey(1)

        combined_transmission = (transmissionB + transmissionG) / 2.0

        # Normalize the combined transmission map
        combined_transmission_normalized = (combined_transmission * 255).astype(np.uint8)

        # Save the combined transmission map as an output image
        output_folder = "output"
        output_path = os.path.join(path, output_folder)
        os.makedirs(output_path, exist_ok=True)
        combined_transmission_output_file = os.path.join(output_path, f"{prefix}_combined_transmission.jpg")
        cv2.imwrite(combined_transmission_output_file, combined_transmission_normalized)
        # Normalize the transmission maps
        transmissionB_norm = transmissionB_image.astype(np.float32) / 255.0
        transmissionG_norm = transmissionG_image.astype(np.float32) / 255.0

        # Combine the transmission maps using a simple average
        combined_transmission = (transmissionB_norm + transmissionG_norm)

        # Scale the combined transmission map back to the range of 0-255
        combined_transmission = (combined_transmission * 255).astype(np.uint8)

        # Display the combined transmission map
        cv2.imshow("Combined Transmission", combined_transmission)

        output_folder = "/Users/achintya/Desktop/UNTV/output"
        output_path = os.path.join(path, output_folder)
        os.makedirs(output_path, exist_ok=True)

        transmission_map_both = np.dstack((transmissionB, transmissionG))
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
        sceneRadiance = OptimalParameter(sceneRadiance)


        stacked_frame = np.hstack((img, sceneRadiance))
        cv2.imshow("Input Image | Output Image", stacked_frame)
        cv2.waitKey(0)


Endtime = datetime.datetime.now()
Time = Endtime - starttime
print('Time', Time)

