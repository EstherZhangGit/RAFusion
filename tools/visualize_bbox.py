import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet, PoseRefineNet
from lib.utils import cloud_to_dims, iterative_points_refine
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default='', help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--output_dir', type=str, default='./output', help='directory to save visualized images')
opt = parser.parse_args()

# Ensure output directory exists
os.makedirs(opt.output_dir, exist_ok=True)


class Visualizer(object):
    def __init__(self, objlist, mesh_points, list_obj, list_rgb, cam_info, point_scale=1.):
        self.objlist = objlist
        self.pt = mesh_points
        self.list_obj = list_obj
        self.list_rgb = list_rgb
        self.point_scale = point_scale  # point cloud unit in mm, point_scale = 1000.
        self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = cam_info[0], cam_info[1], cam_info[2], cam_info[3]
        self.list_dims = self.compute_obj_dim()
        self.cur_r = np.array([0., 0., 0., 1.])
        self.cur_t = np.zeros([3])

    def compute_obj_dim(self):
        list_dims = {}
        for i in self.objlist:
            model_points = self.pt[i] / self.point_scale
            obj_dims = cloud_to_dims(model_points)
            list_dims[i] = obj_dims
        return list_dims

    def update_transformation(self, new_r, new_t):
        self.cur_r = new_r
        self.cur_t = new_t

    def transform_points(self, bbox_points, new_r=None, new_t=None):
        if new_r is None:
            new_r = self.cur_r
        if new_t is None:
            new_t = self.cur_t
        my_t = new_t
        my_r = quaternion_matrix(new_r.copy())[:3, :3]
        #         print("my_r:", my_r)
        #         print("my_t:", my_t)
        pred_bbox = np.dot(bbox_points, my_r.T) + my_t  # 变换点
        return pred_bbox

    def draw_bbox(self, bbox_pxls, img, connected_idxs, color):
        for i, vertices in enumerate(connected_idxs):
            img = cv2.line(img, tuple(bbox_pxls[vertices[0]]), tuple(bbox_pxls[vertices[1]]),
                           color=color, thickness=1)

    def visualize_item(self, index, target, target_r, target_t):

        img = cv2.imread(self.list_rgb[index])
        obj = self.list_obj[index]
        return self.visualize_img(img, obj, target, target_r, target_t)

    def visualize_img(self, img, obj, target, target_r, target_t):
        obj_bbox = self.list_dims[obj]['bbox']
        target_bbox = obj_bbox
        obj_bbox = self.transform_points(obj_bbox)
        obj_bbox_pxls = self.project_point_pxl(obj_bbox)
        #         for px in obj_bbox_pxls: # 预测点云
        #             img = cv2.circle(img, tuple(px), radius=1, color=(255, 0, 0), thickness=2)
        connected_idxs = self.list_dims[obj]['connected_idxs']
        #         if target is not None:
        #             for px in target:# 真实点云
        #                 img = cv2.circle(img, tuple(px), radius=1, color=(0, 0, 255), thickness=1)
        # target_bbox = obj_bbox
        target_bbox = np.dot(target_bbox, target_r.T) + target_t
        target_bbox_pxls = self.project_point_pxl(target_bbox)
        self.draw_bbox(target_bbox_pxls, img, connected_idxs, color=(0, 255, 0))  # 绿色表示真实包围盒
        self.draw_bbox(obj_bbox_pxls, img, connected_idxs, color=(255, 0, 0))  # 蓝色表示预测包围盒
        return img

    def project_point_pxl(self, points):
        points = np.asarray(points)
        cam_matrix = np.eye(3)
        cam_matrix[0, 0] = self.cam_fx
        cam_matrix[1, 1] = self.cam_fy
        cam_matrix[0, 2] = self.cam_cx
        cam_matrix[1, 2] = self.cam_cy

        pixel_points, _ = cv2.projectPoints(points.reshape(1, -1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            cam_matrix, None)

        return np.floor(pixel_points.reshape((-1, 2))).astype(int)


class PoseDataset_visualize(PoseDataset_linemod, Visualizer):
    def __init__(self, mode, num_pointcloud, add_noise, root, noise_trans, refine, objlist):
        PoseDataset_linemod.__init__(self, mode, num_pointcloud, add_noise, root, noise_trans, refine, objlist)
        cam_info = [self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy]
        Visualizer.__init__(self, self.objlist, self.pt, self.list_obj, self.list_rgb, cam_info, point_scale=1000.)

    def get_target(self, index):
        # 调用父类的 __getitem__ 方法，获取包含 target 的数据
        _, _, _, target, _, _ = self.__getitem__(index)
        return target


def main():
    num_objects = 13
    num_points = 500
    objlist = [9]
    iteration = 4
    bs = 1
    testdataset = PoseDataset_visualize('eval', num_points, False, opt.dataset_root, 0.0, True, objlist)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=bs, shuffle=False, num_workers=10)

    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    # 遍历所有样本
    for index in range(len(testdataset)):
        points, choose, img, target, model_points, idx, target_r, target_t = testdataloader.dataset[index]

        # 检查图像是否为 None
        if img is None:
            print(f"Image at index {index} could not be loaded, skipping...")
            continue

        # 检查图像的尺寸是否正确
        if len(img.shape) != 3 or img.shape[0] != 3:  # 这里检查通道数
            print(f"Image at index {index} has invalid dimensions {img.shape}, skipping...")
            continue

        points, choose, img, target, model_points, idx = Variable(points.unsqueeze(0)).cuda(), \
            Variable(choose.unsqueeze(0)).cuda(), \
            Variable(img.unsqueeze(0)).cuda(), \
            Variable(target.unsqueeze(0)).cuda(), \
            Variable(model_points.unsqueeze(0)).cuda(), \
            Variable(idx.unsqueeze(0)).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        _, my_r, my_t = iterative_points_refine(refiner, points, emb, idx, iteration, my_r, my_t, bs, num_points)

        testdataset.update_transformation(my_r, my_t)
        target = target.cpu().detach().numpy()
        target_pxl = testdataset.project_point_pxl(target)
        target_t = target_t / 1000.0

        # 可视化并保存图像
        visualized_img = testdataset.visualize_item(index, target_pxl, target_r, target_t)
        output_path = os.path.join(opt.output_dir, f'visualization_{index:05d}.png')
        cv2.imwrite(output_path, visualized_img)
        print(f"Visualized image saved at: {output_path}")


if __name__ == '__main__':
    main()


