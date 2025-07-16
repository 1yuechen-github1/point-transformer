import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    crop_percentage = 0.3
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

        # 随机裁剪背景部分
    # ----------------------------------------------
    if crop_percentage > 0:
        coord_min = np.min(coord, axis=0)
        coord_max = np.max(coord, axis=0)
        # 计算裁剪区域
        crop_range = (coord_max - coord_min) * crop_percentage
        # 随机选择裁剪区域的偏移
        random_offset = np.random.uniform(low=0, high=crop_range, size=(3,))
        # 生成布尔索引，确保 coord、feat 和 label 一致地被裁剪
        mask = (coord[:, 0] > (coord_min[0] + random_offset[0])) & \
               (coord[:, 1] > (coord_min[1] + random_offset[1])) & \
               (coord[:, 2] > (coord_min[2] + random_offset[2]))
        # 通过布尔索引裁剪数据
        coord = coord[mask]
        feat = feat[mask]
        label = label[mask]
    # ----------------------------------------------

    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
