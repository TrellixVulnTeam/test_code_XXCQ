#!/usr/bin/python

import sys

import numpy as np
import open3d as o3d
import os.path
from basic import bcolors as bcolors

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # Set background color as black
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 2 and argc != 3:
        print(bcolors.WARNING +
              "Usage: pointcloud_viewer(p) input-ply/pcd/xyz/xyzrgb/xyzn/pts" + bcolors.ENDC)
        sys.exit()
    modelname = sys.argv[1]
    if os.path.isfile(modelname) == False:
        print(bcolors.FAIL +
              "ERROR: File {0} doesn't exist!".format(modelname) + bcolors.ENDC)
        sys.exit()
    print("Reading " + modelname)
    idx = modelname.rfind('.')
    suf = modelname[idx + 1:]
    if suf == 'ply' or suf == 'pts' or suf == 'pcd' or suf == 'xyz' or suf == 'xyzrgb' or suf == 'xyzn':
        pcd = o3d.io.read_point_cloud(modelname)
        print(bcolors.OKGREEN, pcd, bcolors.ENDC)
        o3d.visualization.draw_geometries([pcd])
        # custom_draw_geometry(pcd)
    else:
        print(bcolors.FAIL +
              "ERROR: file format {0} is not supported!".format(suf) + bcolors.ENDC)
        sys.exit()
