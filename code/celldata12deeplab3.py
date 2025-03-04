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

    print(f'Current dataset path: {txt_fname}')
    print(f'20, 40, 60, 80 分别代表训练集数据占比!')
    print(f'read_cell_images-> features: {len(features)}, labels: {len(labels)} {"training" if is_train else "validation"} examples read')
    return features, labels





CELL_COLORMAP = [[0, 0, 255],[255, 0, 0], [0, 255, 0],  [255, 0, 255]]


CELL_CLASSES = ['bgd', 'good', 'bad', 'pink']

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

#0907 Evening update for deeplabv3: make all tensors and make each features correspond 1 class of mask
#
#
# class CellDeepLab3SegDataset(torch.utils.data.Dataset):
#     """A customized dataset to load the cell dataset, returning one class mask per call."""
#
#     def __init__(self, is_train, voc_dir):
#         features, labels = read_cell_images(voc_dir, is_train)
#         self.features = features
#         self.labels = labels
#         self.colormap2label = cell_colormap2label()
#         print('read ' + str(len(self.features)) + ' examples')
#
#         # Store all (image index, class index) pairs for efficient access
#         self.image_class_pairs = self._prepare_image_class_pairs()
#
#     def _prepare_image_class_pairs(self):
#         """Prepares a list of tuples where each tuple represents (image_idx, class_idx)."""
#         pairs = []
#         for idx, label in enumerate(self.labels):
#             indexed_label = cell_label_indices(label, self.colormap2label)
#             unique_classes = np.unique(indexed_label)
#             # Exclude background (class 0)
#             for class_idx in unique_classes:
#                 if class_idx != 0:
#                     pairs.append((idx, class_idx))
#         return pairs
#
#     def __getitem__(self, pair_idx):
#         """Returns one class mask and corresponding points for a given image."""
#         # Retrieve the image index and class index for this pair
#         img_idx, class_idx = self.image_class_pairs[pair_idx]
#
#         # Get the corresponding feature (image) and label (segmentation map)
#         feature, label = self.features[img_idx], self.labels[img_idx]
#         indexed_label = cell_label_indices(label, self.colormap2label)
#         feature = feature.to(torch.float)
#         # Create the mask for the selected class
#         mask = (indexed_label == class_idx).to(torch.float)  # Convert mask to float
#
#         # Find points (coordinates) where the mask is true
#         coords = torch.nonzero(mask, as_tuple=False)  # Get coordinates of the class mask
#         if coords.size(0) > 0:
#             # Randomly select a point from within the mask
#             random_point_idx = torch.randint(0, coords.size(0), (1,)).item()
#             chosen_point = coords[random_point_idx]
#             points = torch.tensor([[chosen_point[1].item(), chosen_point[0].item()]])  # [x, y] format
#         else:
#             points = torch.tensor([[-1, -1]])  # No points available, default
#
#         # Return the feature (image), mask (as float), points, and class ID
#         return torch.tensor(feature), mask, points, torch.tensor([class_idx])
#
#     def __len__(self):
#         """Returns the total number of (image, class) pairs."""
#         return len(self.image_class_pairs)
#


"""
0908:
修改 CellDeepLab3SegDataset类，每次返回完整的图像和所有类别的掩码，而不是逐个类别返回掩码。
模型可以在一个 forward pass 中同时处理多个类别，从而提高训练效率，并确保所有类别都能参与分割。
"""
class CellDeepLab3SegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the cell dataset, returning full image and mask with all classes."""

    def __init__(self, is_train, voc_dir):
        features, labels = read_cell_images(voc_dir, is_train)
        self.features = features
        self.labels = labels
        self.colormap2label = cell_colormap2label()  # 颜色到类别ID的映射
        print(f'Read {len(self.features)} examples.')

    def __getitem__(self, idx):
        """Returns the image and full segmentation mask (with all classes) for a given image."""
        # 获取图像和对应的标签
        feature, label = self.features[idx], self.labels[idx]
        indexed_label = cell_label_indices(label, self.colormap2label)


        # 将图像转换为float32类型的tensor
        # If `feature` is already a tensor, ensure it's in float32 format
        if isinstance(feature, torch.Tensor):
            feature = feature.to(torch.float32)
        else:
            # Convert numpy array or other types to float32 tensor
            feature = torch.tensor(feature, dtype=torch.float32)

        # If `indexed_label` is already a tensor, ensure it's in long format
        if isinstance(indexed_label, torch.Tensor):
            mask = indexed_label.to(torch.long)
        else:
            # Convert numpy array or other types to long tensor
            mask = torch.tensor(indexed_label, dtype=torch.long)

        return feature, mask

    def __len__(self):
        """Returns the total number of images."""
        return len(self.features)


def load_data_cell_forsam2(batch_size, test=False):
    """Load the cell dataset."""
    """
    if test: False  use real dataset
    if test: True  use test dataset(small dataset)
    """

    base_dir = ""  # 数据集路径
    base_dir = "/home/ipsdb/"
    cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')



    if test:
        # cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1_test')
        cell_dir = os.path.join(base_dir, 'data/ips/test1024_small')


    print(f'load_data_cell_forsam2->test = {test} batch_size: {batch_size} \n  Current directory: {cell_dir}')

    # num_workers = 0 if os.name == 'nt' else 4
    num_workers = 4
    #0821 add , pin_memory=True to minimize CPU overhead if transfering data to GPU
    train_iter = torch.utils.data.DataLoader(CellDeepLab3SegDataset(True, cell_dir),
                                             batch_size, shuffle=True, drop_last=True, num_workers=num_workers,  pin_memory=True)
    val_iter = torch.utils.data.DataLoader(CellDeepLab3SegDataset(False, cell_dir),
                                           batch_size, drop_last=True, num_workers=num_workers,  pin_memory=True)
    return train_iter, val_iter
