import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch.nn.functional as F
import PIL.Image as Image
import random


class Dataset360N(Dataset):
    def __init__(self, filenames_filepath, delimiter, input_shape):
        self.length = 0
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.filenames_filepath = filenames_filepath
        self.delim = delimiter
        self.data_paths = {}
        self.init_data_dict()
        self.gather_filepaths()

    def init_data_dict(self):
        self.data_paths = {
            "rgb": [],
            "surface": []
        }
    
    def gather_filepaths(self):
        fd = open(self.filenames_filepath, 'r')
        lines = fd.readlines()
        for line in lines:
            splits = line.split(self.delim)
            self.data_paths["rgb"].append(splits[0])
            # TODO: check for filenames files format
            self.data_paths["surface"].append(splits[6])
        fd.close()
        assert len(self.data_paths["rgb"]) == len(self.data_paths["surface"])
        self.length = len(self.data_paths["rgb"])

    def load_rgb(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 3), dtype = np.float32)
        rgb_np = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
        return rgb_np

    def load_float(self, filepath):
        if not os.path.exists(filepath):
            print("\tGiven filepath <{}> does not exist".format(filepath))
            return np.zeros((self.height, self.width, 3), dtype = np.float32)
        surface_np = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        # Creates mask for invalid values
        surface_np[np.isnan(surface_np)] = 0.0
        mask_np = np.ones_like(surface_np)
        mask_np[np.sum(surface_np, 2) == 0.0] = 0.0
        return surface_np, mask_np

    def clean_normal(self, normal):
        # check if normals are close to the dominant
        # coord system normals
        shape = normal.shape
        vecs = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0 ,1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]
        for vec in vecs:
            vec_mat = np.asarray(vec, dtype = np.float32)
            vec_mat = np.expand_dims(vec_mat, 0)
            vec_mat = np.expand_dims(vec_mat, 1)
            vec_mat = vec_mat.repeat(shape[0], 0)
            vec_mat = vec_mat.repeat(shape[1], 1)
            inds = np.isclose(normal, vec_mat, 0.0001, 0.1)
            inds = inds[:, :, 0] & inds[:, :, 1] & inds[:, :, 2]
            normal[inds, 0] = vec[0]
            normal[inds, 1] = vec[1]
            normal[inds, 2] = vec[2]
        return normal

    def make_tensor(self, np_array):
        np_array = np_array.transpose(2, 0, 1)
        tensor = torch.from_numpy(np_array)
        return torch.as_tensor(tensor, dtype = torch.float32)

    def load_item(self, idx):
        item = { }
        if (idx >= self.length):
            print("Index out of range.")
        else:
            rgb_np = self.load_rgb(self.data_paths["rgb"][idx])
            surface_np, mask_np = self.load_float(self.data_paths["surface"][idx])
            surface_np = self.clean_normal(surface_np)
            rgb = self.make_tensor(rgb_np)
            surface = self.make_tensor(surface_np)
            surface = F.normalize(surface, p = 2, dim = 1)
            mask = self.make_tensor(mask_np)
            item['input_rgb'] = rgb
            item['target_surface'] = surface
            item['mask'] = mask
            item['filename'] = os.path.basename(self.data_paths["surface"][idx])
            return item
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.load_item(idx)