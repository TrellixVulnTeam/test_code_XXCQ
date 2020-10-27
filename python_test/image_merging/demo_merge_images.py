import os
import cv2
import numpy as np

cnt = 0
names = []
i = 0
folder = 'chuxi2'
for filename in os.listdir(folder):
    # print(filename)
    names.append(folder + '/' + filename)
    i = i + 1
    if i == 5:
        img0 = cv2.imread(names[0])
        img1 = cv2.imread(names[1])
        img2 = cv2.imread(names[2])
        img3 = cv2.imread(names[3])
        img4 = cv2.imread(names[4])
        vis = np.concatenate((img0, img1, img2, img3, img4), axis=1)
        # img1 = cv2.imread(names[0])
        # img2 = cv2.imread('ia_500000114.jpg')
        # img3 = cv2.imread('ia_500000115.jpg')
        # img4 = cv2.imread('ia_500000116.jpg')
        # img5 = cv2.imread('ia_500000117.jpg')
        # vis = np.concatenate((img1, img2, img3, img4, img5), axis=1)
        cv2.imwrite(folder + '/out' + str(cnt) + '.jpg', vis)
        names = []
        i = 0
        cnt = cnt + 1

        
    # if filename.endswith(".asm") or filename.endswith(".py"): 
    #      # print(os.path.join(directory, filename))
    #     continue
    # else:
    #     continue