"""
This script Generates Images from the pixel values mentioned in the train CSV file.
"""
import numpy as np
import pandas as pd
import cv2


train_data = pd.read_csv("..//csv_files//datasets//train.csv")

print("Script Started......")


def train_csv_to_image(series):
    pixels = series["pixels"]
    pixels = np.array(pixels.split())
    pixels = pixels.reshape(48, 48)
    pixels = pixels.astype('uint8')
    return pixels


count = 0
for i in range(1, train_data.shape[0]):
    face = train_data.iloc[i]
    img = train_csv_to_image(face)
    if face['emotion'] == 0:
        path = "..//Expression_Detector_Dataset//train//0-angry//"
        emotion = 'Angry'
        count += 1
        cv2.imwrite(path+emotion+str(count)+'.jpg', img)
    elif face['emotion'] == 1:
        path = "..//Expression_Detector_Dataset//train//1-disgust//"
        emotion = 'Disgust'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)
    elif face['emotion'] == 2:
        path = "..//Expression_Detector_Dataset//train//2-fear//"
        emotion = 'Fear'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)
    elif face['emotion'] == 3:
        path = "..//Expression_Detector_Dataset//train//3-happy//"
        emotion = 'Happy'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)
    elif face['emotion'] == 4:
        path = "..//Expression_Detector_Dataset//train//4-sad//"
        emotion = 'Sad'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)
    elif face['emotion'] == 5:
        path = "..//Expression_Detector_Dataset//train//5-surprise//"
        emotion = 'Surprise'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)
    else:
        path = "..//Expression_Detector_Dataset//train//6-Neutral//"
        emotion = 'Neutral'
        count += 1
        cv2.imwrite(path + emotion + str(count) + '.jpg', img)

print("Training Set Generated")
