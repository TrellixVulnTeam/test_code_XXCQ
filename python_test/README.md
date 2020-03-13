# python_test

当前路径下，有几个文件是基于 Open3D 实现的 3D model viewer 工具：
- `basic.py`：用于放置一些基础代码供其他多个文件使用；
- `mesh_viewer.py`：用于 preview mesh，支持 ply 和 obj 格式；
- `pointcloud_viewer.py`：用于 preview point cloud，支持 ply, pts, xyz, xyzn, xyzrgb 等多种点云格式；

建议在 `~/.zshrc` 中定义类似如下的 alias：
```shell
## Some custom command definition to preview 3D models
# for mesh
alias mesh_viewer="python ~/dev/test_code/python_test/mesh_viewer.py"
alias m="mesh_viewer"
# for point cloud
alias pointcloud_viewer="python ~/dev/test_code/python_test/pointcloud_viewer.py"
alias p="pointcloud_viewer"
```

这样就能用非常简单的命令 `m` 和 `p` 来直接从终端来预览模型了：
```shell
m some_mesh.ply
p some_point_cloud.pts
```