"""
This script Generates Images from the pixel values mentioned in the test CSV file.
"""
import numpy as np
import pandas as pd
import cv2

test_data = pd.read_csv("..//csv_files//datasets//test.csv")

print("Script Started......")


def csv_to_image(series):
    pixels = series["pixels"]
    pixels = np.array(pixels.split())
    pixels = pixels.reshape(48, 48)
    pixels = pixels.astype('uint8')
    return pixels


count = 0
for i in range(1, test_data.shape[0]):
    face = test_data.iloc[i]
    img = csv_to_image(face)
    path = "..//Expression_Detector_Dataset//test//"
    count += 1
    cv2.imwrite(path + "Test" + str(count) + '.jpg', img)


print("Test/Validation Set Generated")
