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

#0819 Evening update for SAM2-points, masks dictionary
class CellSAM2SegDataset(torch.utils.data.Dataset):
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

        inds = np.unique(indexed_label)
        # print(f"__getitem__ idx: {idx}, inds: {inds}")

        # Check if the image only contains background
        if len(inds) == 1 and inds[0] == 0:
            # If only background, return default values or empty data
            return feature, {}, [], torch.zeros(1, 1)

        points = []
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
                points.append([[chosen_point[1].item(), chosen_point[0].item()]])  # [x, y] format

        return feature, masks, points, torch.ones(len(masks), 1)

    def __len__(self):
        return len(self.features)


def custom_collate(batch):
    # Filter out None or empty batches
    batch = [item for item in batch if item is not None]

    # Stack features and remove the batch dimension
    features = torch.stack([item[0] for item in batch])
    features = features.squeeze(1).numpy()  # Convert shape [batch_size, 3, 480, 480] to [batch_size, 3, 480, 480]
    features = np.transpose(features, (0, 2, 3, 1))  # Convert shape to [batch_size, 480, 480, 3]


    masks = [item[1] for item in batch]
    points = [item[2] for item in batch]
    ones = [item[3] for item in batch]

    return features, masks, points, ones


def load_data_cell_forsam2(batch_size, test=False):
    """Load the cell dataset."""
    """
    if test: False  use real dataset
    if test: True  use test dataset(small dataset)
    """

    base_dir = "/home/ipsdb/"
    cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')
    # cell_dir = os.path.join(base_dir, 'data/ips/outpatches_1024')
    cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')  #20250216, for fair comparation, use the same dataset with Dice
    if test:
        # cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1_test')
        cell_dir = os.path.join(base_dir, 'data/ips/test1024_small')


    print(f'Current directory: {cell_dir}')

    # num_workers = 0 if os.name == 'nt' else 4
    num_workers = 4
    #0821 add , pin_memory=True to minimize CPU overhead if transfering data to GPU
    train_iter = torch.utils.data.DataLoader(CellSAM2SegDataset(True, cell_dir),
                                             batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)
    val_iter = torch.utils.data.DataLoader(CellSAM2SegDataset(False, cell_dir),
                                           batch_size, drop_last=True, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)
    return train_iter, val_iter
