import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import rasterio
import json
from torchvision import transforms


def find_dis(points):
    square = np.sum(points*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(points, points.T) + square[None, :], 0.0))
    if len(dis) > 3:
        dis = np.sum(np.partition(dis, 3, axis=1)[:, :4], axis=1) / 3
    else:
        dis = np.sum(dis, axis=1) / max(len(dis) - 1, 1)
    return dis


def calc_inner_areas(bboxes, window_size):
    inner_lft = np.maximum(0.0, bboxes[:, 0])
    inner_up = np.maximum(0.0, bboxes[:, 1])
    inner_rht = np.minimum(window_size, bboxes[:, 2])
    inner_down = np.minimum(window_size, bboxes[:, 3])
    return np.maximum(inner_rht - inner_lft, 0.0) * np.maximum(inner_down - inner_up, 0.0)


class Tree(data.Dataset):
    def __init__(self,
                 image_uri,
                 annotate_uri,
                 indices,
                 method,
                 window_size,
                 stride,
                 augment,
                 ):
        self.image_uri = image_uri
        self.df_pt = pd.read_csv(annotate_uri)
        self.indices = indices
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        if method not in ('train', 'val', 'test'):
            raise ValueError('method must be "train", "val", or "test"')
        self.method = method

        with rasterio.open(self.image_uri) as image:
            self.patch_shape = _get_patch_shape(image.shape, self.window_size,
                                                self.stride)

        if self.indices is None:
            self.indices = list(range(
                self.patch_shape[0] * self.patch_shape[1]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row, col = np.unravel_index(self.indices[i], self.patch_shape)
        row *= self.stride
        col *= self.stride

        # print(row + self.window_size, col + self.window_size)
        with rasterio.open(self.image_uri) as image:
           X = image.read(window=((row, row + self.window_size),
                                  (col, col + self.window_size))).astype(np.float32)
           # X = X[:3]
           # X = X / 256 - .5
           X -= X.mean((1, 2), keepdims=True)
           X /= X.std((1, 2), keepdims=True)
        pts = self.df_pt[
            self.df_pt['row'].between(row, row + self.window_size - 1) &
            self.df_pt['col'].between(col, col + self.window_size - 1)].values

        if self.method == 'val':
            return torch.from_numpy(X), len(pts), "patch-%d" % self.indices[i]

        dis = find_dis(pts)
        nearest_dis = np.clip(dis, 4.0, 128.0)

        pts_lft_up = pts - nearest_dis[:, None] / 2.0
        pts_rht_down = pts + nearest_dis[:, None] / 2.0
        bboxes = np.concatenate((pts_lft_up, pts_rht_down), axis=1)
        bboxes -= [row, col, row, col]

        inner_areas = calc_inner_areas(bboxes, self.window_size)
        original_areas = nearest_dis * nearest_dis
        ratios = np.clip(1.0 * inner_areas / original_areas, 0.0, 1.0)

        mask = ratios >= 0.3
        target = ratios[mask]
        pts = pts[mask]

        return (torch.from_numpy(X),
                torch.from_numpy(pts.copy()).float(),
                torch.from_numpy(target.copy()).float(),
                self.window_size
                )


def _get_patch_shape(image_shape, window_size, stride):
    return (((image_shape[0] - window_size) // stride + 1),
            ((image_shape[1] - window_size) // stride + 1))


def get_datasets(image_uri, annotate_uri, split_file, augment=True):
    with open(split_file) as f:
        split_info = json.load(f)
    window_size = split_info['window_size']
    stride = split_info['stride']

    train_dataset = Tree(image_uri,
                         annotate_uri,
                         split_info['train_indices'],
                         'train',
                         window_size,
                         stride,
                         augment,
                         )
    val_dataset = Tree(image_uri,
                       annotate_uri,
                       split_info['val_indices'],
                       "val",
                       window_size,
                       stride,
                       False,
                       )
    return train_dataset, val_dataset
