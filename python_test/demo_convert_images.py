import cv2
import os
import sys
import argparse

# This code shows how to convert all RGBD images in a folder with jpg (color) and png (depth)
# format to ppm and pgm image format.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This code converts all RGBD images in a folder with jpg (color) and png (depth) format to ppm and pgm image format.
    ''')
    parser.add_argument('rgbd_folder', type=str, default='',
                        help='rgbd folder (BundleFusion dataset)')
    args = parser.parse_args()
    rgbd_folder = args.rgbd_folder

    # Create an output folder if not exists
    out_folder = rgbd_folder + '_ppm_pgm'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    lst = os.listdir(rgbd_folder)
    lst.sort()  # sort all files to traverse them in alphabetical order
    color_name = ''
    for filename in lst:
        # print(filename)
        # Check prefix and suffix to ensure current file is the correct one
        if filename.startswith('frame-'):
            if filename.endswith('.color.jpg'):
                print("Checking color image ", filename)
                color_name = filename
                continue
            elif filename.endswith('.depth.png'):
                print("Checking depth image ", filename)
                depth_name = filename

                # The following is the way to call another python script
                command = 'python3 rgbd_jpg_png_to_ppm_pgm.py ' + \
                    rgbd_folder + '/' + color_name + ' ' + rgbd_folder + \
                    '/' + depth_name + ' -d ' + out_folder
                # print(command)
                os.system(command)
