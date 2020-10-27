import cv2
import os
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This code will convert an input color PPM image to JPG format and convert depth PGM image to PNG format.
    ''')
    parser.add_argument('color_file', type=str, default='',
                        help='input color image')
    parser.add_argument('depth_file', type=str, default='',
                        help='input depth image')
    parser.add_argument("-d", '--out_dir', type=str, default='',
                        help='output image directory')
    args = parser.parse_args()

    color_image = cv2.imread(args.color_file, cv2.IMREAD_UNCHANGED)
    # Use 'try-except' to check if an image is read successfully or not
    try:
        color_image.shape
        print(color_image.shape, color_image.dtype)
    except:
        print('Cannot read color image file: ', args.color_file)
        sys.exit(1)

    depth_image = cv2.imread(args.depth_file, cv2.IMREAD_UNCHANGED)
    try:
        depth_image.shape
        print(depth_image.shape, depth_image.dtype)
    except:
        print('Cannot read depth image file: ', args.depth_file)
        sys.exit(1)

    # cv2.imshow('image', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    color_out_name = ''
    if args.out_dir:
        base = os.path.basename(args.color_file)  # get 'name.jpg'
        name = os.path.splitext(base)[0]  # get 'name' (while [1] is '.jpg')
        color_out_name = args.out_dir + '/' + name + '.jpg'
    else:
        color_name = os.path.splitext(args.color_file)[0]  # get '.../dir/name'
        color_out_name = color_name + '.jpg'
    print("Save color image: ", color_out_name)
    cv2.imwrite(color_out_name, color_image)

    depth_out_name = ''
    if args.out_dir:
        base = os.path.basename(args.depth_file)
        name = os.path.splitext(base)[0]
        depth_out_name = args.out_dir + '/' + name + '.png'
    else:
        depth_name = os.path.splitext(args.depth_file)[0]
        depth_out_name = depth_name + '.png'
    print("Save depth image: ", depth_out_name)
    cv2.imwrite(depth_out_name, depth_image)
