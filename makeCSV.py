#96x96
import numpy as np
import os
from PIL import Image
import csv

#학습할 표정 0~6
FaceShot = 6

#Image resize 96x96 & convert to grayscale
#path = 'pic' + str(FaceShot)
path = str(FaceShot)

imagePaths = [os.path.join(path,file_name) for file_name in os.listdir(path)]
global AllData
start = 1

for imagePath in imagePaths:
    img = Image.open(imagePath).convert('L')
    resize_image = img.resize((96, 96))
    img_numpy = np.array(resize_image)
    img_numpy = np.append(img_numpy, FaceShot)

    if start:
        AllData = img_numpy.flatten()
        start = 0
    else:
        flatten_numpy = img_numpy.flatten()
        AllData = np.vstack([flatten_numpy, AllData])

final = 'TrainData' + str(FaceShot) + '.csv'
with open(final, 'w', newline='') as f:
    wrtr = csv.writer(f)
    wrtr.writerows(AllData)


