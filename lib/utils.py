import logging
import numpy as np
import open3d as o3d
import copy

import torch
from torch.autograd import Variable
from transformations import quaternion_matrix, quaternion_from_matrix
from collections import defaultdict

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l


def launchTensorBoard(data_read_dir):
    # This is nicer but does not show scalars...
    # from tensorboard import default
    from tensorboard import program
    tb = program.TensorBoard()
    # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    port = 6006
    while True:
        # tb.configure(argv=[None, '--logdir', self.path, '--port', str(port)])
        tb.configure(argv=[None, '--logdir', data_read_dir, '--port', str(port)])
        # tb.configure(argv=[None, '--logdir', self.data_read_dir, '--port', str(port), '--embeddings_data', 'None'])
        try:
            url = tb.launch()
            break
        except Exception as e:
            print(e)
        port += 2
        if port > 7000:
            print("Could not find an open port for tensorboard. Tensorboard has to be launched manually.")
            return
    print("Tensorboard running at {}".format(url))


def cloud_to_dims(obj_cloud):

    obj_cloud = copy.deepcopy(obj_cloud)
    centroid = np.array([
        (np.amax(obj_cloud[:, 0]) + np.amin(obj_cloud[:, 0])) / 2,
        (np.amax(obj_cloud[:, 1]) + np.amin(obj_cloud[:, 1])) / 2,
        (np.amax(obj_cloud[:, 2]) + np.amin(obj_cloud[:, 2])) / 2,
    ])

    object_transform = np.eye(4)
    object_transform[0, 3] = -centroid[0]
    object_transform[1, 3] = -centroid[1]
    object_transform[2, 3] = -centroid[2]

    # Transform our point cloud. I.e. center and rotate to new frame.
    for i in range (obj_cloud.shape[0]):
        obj_cloud[i] = np.dot(object_transform, [obj_cloud[i][0], obj_cloud[i][1], obj_cloud[i][2], 1])[:3]

    width = np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0])
    height = np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1])
    depth = np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2])
    whd = [width, height, depth]

    w_from_center = max(abs(np.amax(obj_cloud[:,0])), abs(np.amin(obj_cloud[:,0])))
    h_from_center = max(abs(np.amax(obj_cloud[:,1])), abs(np.amin(obj_cloud[:,1])))
    d_from_center = max(abs(np.amax(obj_cloud[:,2])), abs(np.amin(obj_cloud[:,2])))
    sdf_whd = [w_from_center, h_from_center, d_from_center]

    bbox, connected_idxs = compute_3d_bbox(centroid, sdf_whd)
    return {'centroid': centroid, 'dim': whd, 'dim_from_centroid': sdf_whd, 'bbox': bbox,
            'connected_idxs': connected_idxs}


def compute_3d_bbox(centroid, dim_from_centroid):
    """
    Compute 3d bounding box of an object, given centroid and dimension form the centroid
    :param centroid:
    :param dim_from_centroid:
    :return: list of 8 corners of bbox
    """
    c = np.asarray(centroid).copy()
    dim = np.asarray(dim_from_centroid).copy()
    bbox = [c for _ in range(8)]
    bbox[:4] -= dim
    bbox[4:] += dim
    for i in range(3):
        bbox[i+1][i] += 2*dim[i]
    for i in range(3):
        bbox[i+5][i] -= 2*dim[i]

    connected_idx = []
    for i in range(5):
        for j in range(1, 8):
            a = bbox[i]
            b = bbox[j]
            if len(set(a).intersection(set(b)))==2:
                connected_idx.append([i, j])

    return bbox, connected_idx


def iterative_points_refine(refiner, points, emb, idx, iteration, my_r, my_t, bs, num_points):
    """
    Refine predicted transformation matrix
    :param refiner: refiner network
    :param points: cropped point cloud
    :param emb: embedding from estimator
    :param idx: output of estimator
    :param iteration: number of refinement iteration
    :param my_r: current pred_r
    :param my_t: current pred_t
    :param bs: batch size
    :param num_points: number of points from mesh
    :return: my_pred, my_r, my_t
    """
    for ite in range (0, iteration):
        T = Variable(torch.from_numpy(my_t.astype (np.float32))).cuda ().view(1, 3).repeat(bs*num_points,1).contiguous().view (1,bs*num_points,3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype (np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)

        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

        return my_pred, my_r, my_t