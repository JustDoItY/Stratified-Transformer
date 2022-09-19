import numpy as np
from util.vis_util import colors
import open3d as o3d

colors = np.array(colors)
def draw_npy(path: str, label: str = None, flag:str = 'origin'):
    # o3d.io.read_point_cloud()
    raw_points = np.load(path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(path.split('/')[-1] +"-"+ flag)
    vis.get_render_option().point_size = 0.1
    vis.get_render_option().background_color = np.asarray([0,0,0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points[:, :3])
    # pcd.paint_uniform_color([1,1,1])
    if label is not None:
        color_index = np.load(label).astype(np.int32).reshape(-1).tolist()
    else:
        color_index = np.array(raw_points[:, 6], dtype=np.int32).tolist()
    pcd.colors = o3d.utility.Vector3dVector(colors[color_index])

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

p = '/home/yjzx/桌面/zanguokuan/dgcnn.pytorch/data/stanford_indoor3d/Area_5_WC_1.npy'

draw_npy(p, 'train_test_5_npy/Area_5_WC_1_19_label.npy', 'pred')