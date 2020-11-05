import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import argparse
import open3d_tutorial as o3dtut

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This script reads a registered pair of color and depth images and generates and views a colored 3D point cloud in the
    PLY format.
    ''')
    parser.add_argument('depth_file', type=str, default='',
                        help='input depth image (format: png)')
    parser.add_argument(
        "-r", '--rgb_file', nargs='?', default='', help='input color image (format: jpg/png). Default: empty')
    parser.add_argument("-s", '--depth_scale', nargs='?', type=float, default=1000,
                        help='depth value scale factor (scale to meter). Default: 1000')
    parser.add_argument("-t", '--depth_trunc', nargs='?', type=float, default=10.0,
                        help='depth values larger than this will be truncated (unit: meter). Default: 10.0')
    parser.add_argument("-p", '--ply_file', nargs='?', default='',
                        help='output PLY file (format: ply). Default: empty')
    parser.add_argument("-i", '--depth_intrinsic', type=str, default='',
                        help='depth image intrinsics file (format: json). Or use input argument fx, fy, cx, cy instead')
    parser.add_argument('--fx', nargs='?', type=float, default=-1,
                        help='depth intrinsics: fx')
    parser.add_argument('--fy', nargs='?', type=float, default=-1,
                        help='depth intrinsics: fy')
    parser.add_argument('--cx', nargs='?', type=float, default=-1,
                        help='depth intrinsics: cx')
    parser.add_argument('--cy', nargs='?', type=float, default=-1,
                        help='depth intrinsics: cy')
    parser.add_argument('--width', nargs='?', type=int, default=-1,
                        help='depth intrinsics: image width')
    parser.add_argument('--height', nargs='?', type=int, default=-1,
                        help='depth intrinsics: image height')
    args = parser.parse_args()

    print(o3d.__version__)

    print("Read depth image file: ", args.depth_file)
    depth_raw = o3d.io.read_image(args.depth_file)

    intrinsics = None
    if args.depth_intrinsic:
        print("Read depth intrinsic file: ", args.depth_intrinsic)
        intrinsics = o3d.io.read_pinhole_camera_intrinsic(args.depth_intrinsic)
    elif args.fx > 0 and args.fy > 0 and args.cx > 0 and args.cy > 0 and args.width > 0 and args.height > 0:
        print("Depth intrinsics:", args.fx, args.fy,
              args.cx, args.cy, args.width, args.height)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            args.width, args.height, args.fx, args.fy, args.cx, args.cy)
        # print(intrinsics.intrinsic_matrix, intrinsic_matrix.width, intrinsic_matrix.height)
    else:
        parser.print_help()
        sys.exit(1)

    pcd = None
    if args.rgb_file:
        print("Read color image file: ", args.rgb_file)
        color_raw = o3d.io.read_image(args.rgb_file)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=args.depth_scale, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
        print(rgbd_image)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsics)

        # # Use default intrinsic parameters
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd_image,
        #     o3d.camera.PinholeCameraIntrinsic(
        #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_raw, intrinsics, depth_scale=args.depth_scale, depth_trunc=args.depth_trunc)

    print(pcd)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd], zoom=0.5)

    if args.ply_file:
        print("Save point cloud to file: ", args.ply_file)
        o3d.io.write_point_cloud(args.ply_file, pcd)
