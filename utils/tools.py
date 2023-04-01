import numpy as np
import matplotlib.pyplot as plot
import open3d as o3d

def estimate_normals(points, K_for_normal_estimation=10):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=K_for_normal_estimation),fast_normal_computation=False)
    o3d.geometry.PointCloud.orient_normals_to_align_with_direction(pc, orientation_reference=np.array([0.0, 0.0, 1.0]))
    pc_normals = np.asarray(pc.normals,dtype=np.float32)
    pc_normals[pc_normals==0.]=1.
    return pc_normals

def search_knns(points, K):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_kd_tree = o3d.geometry.KDTreeFlann(pcd)
    nns = []
    for i in range(len(pcd.points)):
        xyz = pcd.points[i]
        [_,idx,_] = pcd_kd_tree.search_knn_vector_3d(xyz, K+1)
        idx = np.asarray(idx)
        nns.append(idx[1:])
    return np.asarray(nns)  # return K nearest neighbors' index in pcd.points


def vecs_angle(vec1, vec2):
    if np.linalg.norm(vec1)==0:
        vec1 = np.asarray([0.5,0.5,0.5],dtype=np.float64)
    if np.linalg.norm(vec2)==0:
        vec2 = np.asarray([0.5,0.5,0.5],dtype=np.float64)

    value = np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if value > 1:
        return np.arccos(1)
    elif value < -1:
        return np.arccos(-1)
    return np.arccos(value)

def calculate_PPFs(points, normals, nns_idx):
    PPFs = []
    for i in range(len(points)):
        x = points[i]
        x_normal = normals[i]
        nns = nns_idx[i]
        PPF = []
        for n_idx in nns:
            nn = points[n_idx]
            nn_normal = normals[n_idx]
            delta = x - nn
            # value = np.asarray([vecs_angle(x_normal,delta) / np.pi, vecs_angle(nn_normal, delta) / np.pi, vecs_angle(x_normal,nn_normal) / np.pi, np.linalg.norm(delta)])
            value = np.asarray([vecs_angle(x_normal,delta), vecs_angle(nn_normal, delta), vecs_angle(x_normal,nn_normal), np.linalg.norm(delta)])
            value = value / np.linalg.norm(value)   # unitization
            PPF.append(value)
        PPFs.append(np.asarray(PPF))
    return np.asarray(PPFs)

def shuffle(points):
    shuffled_idx = np.random.choice(np.arange(0, len(points)), size=len(points), replace=False)
    shuffled_points = points[shuffled_idx]
    return shuffled_idx,shuffled_points

def features_project(features1, features2):
    return np.matmul(features1, features2)

def vec_softmax(f):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))

def view_deep_graph(deep_graph,title,show_text=False):
    if show_text:
        for i in range(deep_graph.shape[0]):
            for j in range(deep_graph.shape[1]):
                plot.text(j,i,'{:.2f}'.format(deep_graph[i][j]), ha="center", va="center", color="w")
    plot.title(title)
    plot.imshow(deep_graph)
    plot.colorbar()
    plot.tight_layout()
    plot.show()

def view_avg_deep_graph(deep_graph):
    view_deep_graph(deep_graph, title='deep graph for point clouds registration',show_text=True)

def view_src_binarylization_deep_graph(deep_graph):
    binary = np.zeros_like(deep_graph)
    for i in range(deep_graph.shape[0]):
        max_value_idx = np.argmax(deep_graph[i])
        binary[i,max_value_idx]=1
    view_deep_graph(binary,title='row correspondences fliter')

def view_tgt_binarylization_deep_graph(deep_graph):
    binary = np.zeros_like(deep_graph)
    deep_graph = deep_graph.transpose(1,0)
    for j in range(deep_graph.shape[0]):
        max_value_idx = np.argmax(deep_graph[j])
        binary[max_value_idx,j]=1
    view_deep_graph(binary,title='column correspondences fliter')

def visualize_correspondences(src_pc, tgt_pc, correspondences, times=3):
    src_pc = src_pc * times
    tgt_pc = tgt_pc * times
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 0.05
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pc)
    src.paint_uniform_color([0, 1, 0])
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pc)
    tgt.paint_uniform_color([1, 0, 0])

    vis.add_geometry(src)
    vis.add_geometry(tgt)
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.points = o3d.utility.Vector3dVector(np.concatenate((src_pc,tgt_pc)))
    new_correspondences = set()
    for pair in correspondences:
        new_pair=tuple((pair[0],pair[1]+src_pc.shape[0]))
        new_correspondences.add(new_pair)
    new_correspondences = np.asarray(list(new_correspondences))
    lines_pcd.lines = o3d.utility.Vector2iVector(new_correspondences)
    lines_pcd.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6] for _ in range(len(new_correspondences))])
    vis.add_geometry(lines_pcd)
    vis.run()

def visualize_pcr(src_pc, tgt_pc, est_pc, win_name, times=3):
    src_pc = src_pc * times
    tgt_pc = tgt_pc * times
    est_pc = est_pc * times
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win_name)
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pc)
    src.paint_uniform_color([0, 1, 0])
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pc)
    tgt.paint_uniform_color([1, 0, 0])
    est = o3d.geometry.PointCloud()
    est.points = o3d.utility.Vector3dVector(est_pc)
    est.paint_uniform_color([0, 0, 1])
    vis.add_geometry(src)
    vis.add_geometry(tgt)
    vis.add_geometry(est)
    vis.run()