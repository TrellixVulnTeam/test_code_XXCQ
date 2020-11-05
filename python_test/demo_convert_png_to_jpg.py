import cv2
import os
import sys
import argparse

if __name__ == '__main__':
    folder = '/Users/chaowang/dev/rgbd_dataset/iphone/lucy14/raw_rgb'
    lst = os.listdir(folder)
    lst.sort()  # sort all files to traverse them in alphabetical order
    for filename in lst:
        # print(filename)
        color_image = cv2.imread(folder + '/' + filename, cv2.IMREAD_UNCHANGED)
        
        name = os.path.splitext(filename)[0]
        # print(name)
        cv2.imwrite(folder + '/' + name + '.JPG', color_image)
