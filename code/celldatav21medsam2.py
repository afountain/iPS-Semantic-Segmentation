import os

import numpy as np
import torchvision
import torch
from sympy.codegen.ast import continue_
# 0820 V2.0 for SAM2 which requires images as numpy , so here changed custom_collate to return numpy images
#Todo: 0820 evening try to remove normalization and crop function later....

def read_cell_images(cell_dir, is_train=True):
    """Read all cell feature and label images. """
    txt_fname = os.path.join(cell_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(
            os.path.join(cell_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(
            os.path.join(cell_dir, 'SegmentationClass', f'{fname}.png'),mode))
    print(f'read_cell_images-> features: {len(features)}, labels: {len(labels)} {"training" if is_train else "validation"} examples read')
    return features, labels



CELL_COLORMAP = [[0, 0, 255],[255, 0, 0], [0, 255, 0],  [255, 0, 255]]


CELL_CLASSES = ['bgd', 'good', 'bad', 'uncertain']

def cell_colormap2label():
    """Build the mapping from RGB to class labels for cell images."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(CELL_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    return colormap2label

def cell_label_indices(colormap, colormap2label):
    """Map the RGB values to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]

def cell_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class CellSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the cell dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_cell_images(voc_dir, is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = cell_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.type(torch.float16) / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = cell_rand_crop(self.features[idx], self.labels[idx],
                                        *self.crop_size)
        return (feature, cell_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_cell(batch_size, crop_size):
    """Load the cell dataset."""
    base_dir = "/home/ipsdb/"
    cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')

    print(f'Current directory: {cell_dir}')

    # num_workers = 0 if os.name == 'nt' else 4
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(CellSegDataset(True, crop_size, cell_dir),
                                             batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_iter = torch.utils.data.DataLoader(CellSegDataset(False, crop_size, cell_dir),
                                           batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, val_iter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label2image(pred):
    colormap = torch.tensor(CELL_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


#0903 Morning update: make points as dictionary (as masks), so that use the same way to access in training
class CellMedSAM2SegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the cell dataset."""

    def __init__(self, is_train, voc_dir):
        # self.transform = torchvision.transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.crop_size = crop_size
        features, labels = read_cell_images(voc_dir, is_train)
        self.features = features
        self.labels = labels
        self.colormap2label = cell_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')


    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]
        indexed_label = cell_label_indices(label, self.colormap2label)
        #0901 resize input to 1024
        # # print(f'getitem-> feature.shape{feature.shape}') #[3, 512,512]
        # c, h, w = feature.shape
        # if (h, w) != (1024, 1024):
        #     # Resize to (1024, 1024)
        #     feature = F.interpolate(feature, size=(1024, 1024), mode='bilinear', align_corners=False)

        inds = np.unique(indexed_label)
        # print(f"__getitem__ idx: {idx}, inds: {inds}")

        # Check if the image only contains background
        if len(inds) == 1 and inds[0] == 0:
            # If only background, return default values or empty data
            return feature, {}, [], torch.zeros(1, 1)

        points = {}
        masks = {}
        for ind in inds:
            mask = (indexed_label == ind).to(torch.uint8)
            masks[ind] = mask

            # Calculate the coordinates where mask is true
            coords = torch.nonzero(mask, as_tuple=False)  # Use GPU tensor operations
            if coords.size(0) > 0:  # Ensure coords is not empty
                # Randomly select a point within the mask
                idx = torch.randint(0, coords.size(0), (1,)).item()
                chosen_point = coords[idx]
                # points.append([[chosen_point[1].item(), chosen_point[0].item()]])  # [x, y] format
                points[ind] = [[chosen_point[1].item(), chosen_point[0].item()]]

        return feature, masks, points, torch.ones(len(masks), 1)

    def __len__(self):
        return len(self.features)

#0903 morning V1.0: extract class id and batched tensors for masks and points, but the length of each batch is different
# def custom_collate2(batch):
#     batch = [item for item in batch if item is not None]
#     features = torch.stack([item[0].to(torch.float16) for item in batch])
#
#     # Dictionary to hold batched masks and points indexed by classid
#     batched_masks = {}
#     batched_points = {}
#
#     # Accumulate masks and points for each class ID
#     for _, masks, points, _ in batch:
#         for class_id in masks.keys():
#             if class_id not in batched_masks:
#                 batched_masks[class_id] = []
#                 batched_points[class_id] = []
#             batched_masks[class_id].append(masks[class_id].unsqueeze(0))  # Add batch dimension
#             point_tensor = torch.tensor(points[class_id][0], dtype=torch.float32).unsqueeze(0)
#             batched_points[class_id].append(point_tensor)  # Add batch dimension
#
#     # Stack all masks and points for each class_id
#     for class_id in batched_masks:
#         batched_masks[class_id] = torch.cat(batched_masks[class_id], dim=0)  # Concatenate along the batch dimension
#         batched_points[class_id] = torch.cat(batched_points[class_id], dim=0)
#
#     ones = torch.stack([item[3] for item in batch])
#
#     return features, batched_masks, batched_points, ones

import torch.nn.functional as F


def custom_collate2(batch):
    batch = [item for item in batch if item is not None]
    features = torch.stack([item[0].to(torch.float32) for item in batch])

    num_classes = 4  # Fixed number of classes
    max_height, max_width = 1024, 1024  # Adjust these values based on the maximum expected mask sizes
    max_points = 1  # Adjust based on the maximum number of points per class you expect

    # Initialize tensors for masks and points
    batched_masks = {}
    batched_points = {}
    ones_list = []

    for idx, (_, masks, points, ones) in enumerate(batch):
        for cid in range(num_classes):
            if cid not in batched_masks:
                batched_masks[cid] = torch.zeros((len(batch), 1, max_height, max_width), dtype=torch.uint8)
                batched_points[cid] = torch.zeros((len(batch), max_points, 2), dtype=torch.float32)

            if cid in masks:
                mask = masks[cid]
                height_pad = max_height - mask.shape[0]
                width_pad = max_width - mask.shape[1]
                batched_masks[cid][idx, 0, :mask.shape[0], :mask.shape[1]] = F.pad(mask, (0, width_pad, 0, height_pad))

            if cid in points:
                for p in range(min(max_points, len(points[cid][0]))):
                    batched_points[cid][idx, p, :] = torch.tensor(points[cid][0][p], dtype=torch.float16)

        # Pad ones to have length equal to num_classes
        current_length = ones.shape[0]
        if current_length < num_classes:
            padding_size = num_classes - current_length
            padded_ones = F.pad(ones, (0, 0, 0, padding_size), "constant", 0)  # Padding at the end
        else:
            padded_ones = ones

        ones_list.append(padded_ones)

    # Stack all ones tensors to create a single tensor for the batch
    ones = torch.cat(ones_list, dim=0)

    return features, batched_masks, batched_points, ones





def load_data_cell_forsam2_2(batch_size, test=True):
    """Load the cell dataset."""
    """
    if test: False  use real dataset
    if test: True  use test dataset(small dataset)
    """
    print(f'test is: {test}')
    base_dir = ""  # 数据集路径
    if os.path.isdir("/home/zhang"):
        base_dir = f"/home/zhang/cellseg/Uncertainty/UNSURE2021_code/"
    elif os.path.isdir("/home/james/extdisk"):
        base_dir = f"/home/james/extdisk/datasets/Uncertainty/UNSURE2021_code/"
    else:
        raise Exception("No such directory exists")
    cell_dir = os.path.join(base_dir, 'data/ips/outpatches_1024')
    # cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')
    if test:
        cell_dir = os.path.join(base_dir, 'data/ips/test1024_small')


    print(f'Current directory: {cell_dir}')

    # num_workers = 0 if os.name == 'nt' else 4
    num_workers = 4
    #0821 add , pin_memory=True to minimize CPU overhead if transfering data to GPU
    train_iter = torch.utils.data.DataLoader(CellMedSAM2SegDataset(True, cell_dir),
                                             batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=custom_collate2, pin_memory=True)
    val_iter = torch.utils.data.DataLoader(CellMedSAM2SegDataset(False, cell_dir),
                                           batch_size, drop_last=True, num_workers=num_workers, collate_fn=custom_collate2, pin_memory=True)
    return train_iter, val_iter


