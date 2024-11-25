import torch
import numpy as np
# from utils.pc_utils import *

import os
import glob
import h5py
import random
from tqdm import tqdm
import numpy as np
import scipy
import pandas as pd
import torch
from torch.utils.data import Dataset
from diffusion.pc_utils import *
NUM_POINTS = 1024



class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device("cpu"))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(
            torch.device("cuda:%d" % gpu)
        )
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]

    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3
    return out


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        dataset,
        labels=None,
        indices=None,
        num_samples=None,
        callback_get_label=None,
        imb_ratio=None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        if imb_ratio:
            selected_idx = np.random.choice(
                len(label_to_count), int(0.1 * len(label_to_count)), replace=False
            )
            for idx in selected_idx:
                weights[df["label"] == idx] *= imb_ratio
        self.weights = torch.DoubleTensor(weights.tolist())

    def _get_labels(self, dataset):
        return dataset.label_list

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


import open3d as o3d
import numpy as np

def save_point_cloud_as_ply(point_cloud, file_path):
    """
    Save a point cloud as a .ply file.

    Parameters:
    - point_cloud: numpy array or torch tensor of shape (N, 3)
    - file_path: str, the path where the .ply file will be saved
    """
    # Convert the point cloud to a numpy array if it's a torch tensor
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Save the point cloud as a .ply file
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")


class ModelNet40C(Dataset):
    def __init__(self, args, partition):
        super().__init__()
        self.dataset = args.dataset
        self.partition = partition
        self.scenario = getattr(args, "scenario", "")
        self.subsample = getattr(args, "subsample", 2048)

        if len(args.dataset.split("_")) == 1:
            self.corruption = "original"
        elif len(args.dataset.split("_")) == 2:
            self.corruption = "_".join(args.dataset.split("_")[1:])
        else:
            self.corruption = "_".join(args.dataset.split("_")[1:-1])
        if self.corruption != "original":
            assert partition == "test"
            self.severity = args.dataset.split("_")[-1]

        self.rotate = args.rotate if hasattr(args, "rotate") else True

        # augmentation
        if partition in ["train", "train_all"]:
            self.jitter = args.jitter if hasattr(args, "jitter") else True
            self.random_scale = (
                args.random_scale if hasattr(args, "random_scale") else False
            )
            self.random_rotation = (
                args.random_rotation if hasattr(args, "random_rotation") else True
            )
            self.random_trans = (
                args.random_trans if hasattr(args, "random_trans") else False
            )
            self.aug = args.aug if hasattr(args, "aug") else False
        else:
            (
                self.jitter,
                self.random_scale,
                self.random_rotation,
                self.random_trans,
                self.aug,
            ) = (False, False, False, False, False)

        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(
                [
                    "airplane",
                    "bathtub",
                    "bed",
                    "bench",
                    "bookshelf",
                    "bottle",
                    "bowl",
                    "car",
                    "chair",
                    "cone",
                    "cup",
                    "curtain",
                    "desk",
                    "door",
                    "dresser",
                    "flower_pot",
                    "glass_box",
                    "guitar",
                    "keyboard",
                    "lamp",
                    "laptop",
                    "mantel",
                    "monitor",
                    "night_stand",
                    "person",
                    "piano",
                    "plant",
                    "radio",
                    "range_hood",
                    "sink",
                    "sofa",
                    "stairs",
                    "stool",
                    "table",
                    "tent",
                    "toilet",
                    "tv_stand",
                    "vase",
                    "wardrobe",
                    "xbox",
                ]
            )
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        if self.corruption == "original":
            self.pc_list, self.label_list = self.load_modelnet40(
                args.dataset_dir, partition=partition
            )
        else:
            self.pc_list, self.label_list = self.load_modelnet40_c(
                args.dataset_dir, self.corruption, self.severity
            )

        # print dataset statistics
        unique, counts = np.unique(self.label_list, return_counts=True)
        print(
            f"number of {partition} examples in {args.dataset} : {str(len(self.pc_list))}"
        )
        print(
            f"Occurrences count of classes in {args.dataset} {partition} set: {str(dict(zip(unique, counts)))}"
        )

    def load_modelnet40(self, data_path, partition="train"):
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(data_path, f"ply_data_{partition}*.h5")):
            f = h5py.File(h5_name.strip(), "r")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0).squeeze(-1)
        return all_data, all_label

    def load_modelnet40_c(
        self,
        data_path="data/modelnet40_c",
        corruption="cutout",
        severity=1,
        num_classes=40,
    ):
        if corruption == "original":
            data_dir = os.path.join(data_path, f"data_{corruption}.npy")
            all_data = np.load(data_dir)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir).squeeze(-1)
        elif self.scenario == "mixed":
            corruption_list = [
                "background",
                "cutout",
                "density",
                "density_inc",
                "distortion",
                "distortion_rbf",
                "distortion_rbf_inv",
                "gaussian",
                "impulse",
                "lidar",
                "occlusion",
                "rotation",
                "shear",
                "uniform",
                "upsampling",
            ]
            data_dir_list = [
                os.path.join(data_path, f"data_{corruption}_{severity}.npy")
                for corruption in corruption_list
            ]
            all_data_list = [np.load(data_dir) for data_dir in data_dir_list]
            selected_indices = np.random.choice(
                len(corruption_list), len(all_data_list[0]), replace=True
            )
            all_data = []
            for iter_idx, corruption_idx in enumerate(selected_indices):
                pointcloud_norm_curv = all_data_list[corruption_idx][iter_idx]
                pointcloud = pointcloud_norm_curv[:, :3]
                norm_curv = pointcloud_norm_curv[:, 3:]

                # NUM_POINTS for all corruptions
                if pointcloud.shape[0] > NUM_POINTS:
                    pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
                    norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
                    _, pointcloud, norm_curv = farthest_point_sample_np(
                        pointcloud, norm_curv, NUM_POINTS
                    )
                    pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype(
                        "float32"
                    )
                    norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype("float32")
                N = len(pointcloud)
                mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
                mask[N:] = 0
                ind = np.arange(len(pointcloud))
                while len(pointcloud) < NUM_POINTS:
                    chosen = np.arange(N)
                    np.random.shuffle(chosen)
                    chosen = chosen[: NUM_POINTS - len(pointcloud)]
                    pointcloud = np.concatenate(
                        (pointcloud, pointcloud[chosen]), axis=0
                    )
                    ind = np.concatenate((ind, chosen), axis=0)
                    norm_curv = np.concatenate((norm_curv, norm_curv[chosen]), axis=0)

                all_data.append(pointcloud)
            all_data = np.array(all_data)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir).squeeze(-1)
        else:
            data_dir = os.path.join(data_path, f"data_{corruption}_{severity}.npy")
            all_data = np.load(data_dir)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir)  # .squeeze(-1)
            print(all_label.shape)
            if all_label.ndim == 2:
                all_label = all_label.squeeze(-1)

        if self.scenario == "temporally_correlated":
            sorted_indices = np.argsort(all_label)
            all_data = all_data[sorted_indices]
            all_label = all_label[sorted_indices]

        print(f"num_classes: {num_classes}")

        if num_classes == 40:
            return all_data, all_label

        pointda_label_dict = {
            1: 0,  # bathtub
            2: 1,  # bed
            4: 2,  # bookshelf
            23: 3,  # night_stand(cabinet)
            8: 4,  # chair
            19: 5,  # lamp
            22: 6,  # monitor
            26: 7,  # plant
            30: 8,  # sofa
            33: 9,  # table
        }
        pointda_label = [
            1,
            2,
            4,
            8,
            19,
            22,
            23,
            26,
            30,
            33,
        ]  # 1: bathtub, 2: bed, 4: bookshelf, 8: chair, 19: lamp, 22: monitor, 23: night_stand(cabinet), 26: plant, 30: sofa, 33: table
        pointda_indices = np.isin(all_label, pointda_label).squeeze(-1)
        all_data = all_data[pointda_indices, :, :]
        all_label = all_label[pointda_indices, :]
        all_label = np.array([pointda_label_dict[idx] for idx in all_label])
        return all_data, all_label

    def get_label_to_idx(self, args):
        npy_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, "*", "train", "*.npy"))
        )
        label_to_idx = {
            label: idx
            for idx, label in enumerate(
                list(np.unique([_dir.split("/")[-3] for _dir in npy_list]))
            )
        }
        return label_to_idx

    def __getitem__(self, item):
        pointcloud = self.pc_list[item][:, :3]
        label = self.label_list[item]
        mask = np.ones((len(pointcloud), 1)).astype(pointcloud.dtype)
        ind = np.arange(len(pointcloud))

        # if label == 0:
        #     save_point_cloud_as_ply(pointcloud, f'/content/init.ply')

        # identify duplicated points
        if (
            "occlusion" in self.corruption
            or "density_inc" in self.corruption
            or "lidar" in self.corruption
        ):
            dup_points = (
                np.sum(
                    np.power((pointcloud[None, :, :] - pointcloud[:, None, :]), 2),
                    axis=-1,
                )
                < 1e-8
            )
            dup_points[np.arange(len(pointcloud)), np.arange(len(pointcloud))] = False
            if np.any(dup_points):
                row, col = dup_points.nonzero()
                row, col = row[row < col], col[row < col]
                filter = (row.reshape(-1, 1) == col).astype(float).sum(-1) == 0
                row, col = row[filter], col[filter]
                ind[col] = row
                dup = np.unique(col)
                mask[dup] = 0

        if self.rotate:
            pointcloud = scale(pointcloud, "unit_std")
            pointcloud = rotate_pc(pointcloud)
            if self.random_rotation:
                # print('bad bakht')
                pointcloud = random_rotate_one_axis(pointcloud, "z")

        if self.jitter:
            pointcloud = jitter_pointcloud(pointcloud)

        if mask.sum() > self.subsample:
            valid = mask.nonzero()[0]
            pointcloud_ = pointcloud[mask.flatten()[: len(pointcloud)] > 0]
            pointcloud_ = np.swapaxes(np.expand_dims(pointcloud_, 0), 1, 2)
            centroids, pointcloud_, _ = farthest_point_sample_np(
                pointcloud_, None, self.subsample
            )
            pointcloud_ = np.swapaxes(pointcloud_.squeeze(), 1, 0).astype("float32")
            centroids = centroids.squeeze()
            assert len(centroids) == self.subsample
            mask_ = np.zeros_like(mask)
            mask_[valid[centroids]] = 1  # reg줄  subsample 된 것! 나머지는
            assert np.all(mask[mask_ == 1] == 1)
            mask = mask_
            if self.corruption == "original":
                pointcloud = pointcloud[mask.squeeze(-1).astype(bool)]
                mask = mask[mask.squeeze(-1).astype(bool)]
                ind = np.arange(len(pointcloud))

        if self.subsample < 2048:
            valid = mask.nonzero()[0]
            while len(pointcloud) < NUM_POINTS:  # len(mask):# NUM_POINTS:
                np.random.shuffle(valid)
                chosen = valid[: NUM_POINTS - len(pointcloud)]
                pointcloud = np.concatenate(
                    (
                        pointcloud,
                        pointcloud[chosen],
                    ),
                    axis=0,
                )
                mask = np.concatenate((mask, np.zeros_like(mask[chosen])), axis=0)
                ind = np.concatenate((ind, chosen), axis=0)
                assert len(pointcloud) == len(ind)

        return (pointcloud, label, mask, ind)

    def __len__(self):
        return len(self.pc_list)



