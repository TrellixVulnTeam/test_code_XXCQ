#!/usr/bin/python

import sys

import numpy as np
import open3d as o3d
import os.path
from basic import bcolors as bcolors

# The following code achieves the same effect as:
# o3d.visualization.draw_geometries([pcd])
# But add some custom setting such as black backgrond.
def custom_draw_geometry(pcd):
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
              "Usage: mesh_viewer(v) input-ply/obj [trajectory_json]" + bcolors.ENDC)
        sys.exit()
    modelname = sys.argv[1]
    if os.path.isfile(modelname) == False:
        print(bcolors.FAIL +
              "ERROR: File {0} doesn't exist!".format(modelname) + bcolors.ENDC)
        sys.exit()
    print("Reading " + modelname)
    idx = modelname.rfind('.')
    suf = modelname[idx + 1:]
    if suf == 'ply' or suf == "obj":
        mesh = o3d.io.read_triangle_mesh(modelname)
        print(bcolors.OKGREEN, mesh, bcolors.ENDC)
        if mesh.has_vertex_normals() == False:
            mesh.compute_vertex_normals()
        # You can also paint the mesh with custom color like this:
        # mesh.paint_uniform_color([1, 0.706, 0]) # paint gold color
        o3d.visualization.draw_geometries([mesh])
        # custom_draw_geometry(mesh)
    else:
        print(bcolors.FAIL +
              "ERROR: file format {0} is not supported!".format(suf) + bcolors.ENDC)
        sys.exit()
