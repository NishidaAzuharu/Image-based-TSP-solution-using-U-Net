import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import pandas as pd


class TSPDataset(BaseDataset):
    def __init__(self, split):
        self.load_dict = pd.read_pickle(f"resources/{split}_data.pkl")
        self.input_img = self.load_dict["input_img"]
        self.output_img = self.load_dict["output_img"]
        self.node_cords = self.load_dict["num_node"]


    def __getitem__(self, i):
        img = self.input_img[i]
        img = cv2.resize(img, dsize = (256, 256))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = img[np.newaxis, :, :]
        img = 255 - img
        img = img/255
        img = torch.from_numpy(img.astype(np.float32)).clone()

        label_img = self.output_img[i]
        label_img = cv2.resize(label_img, dsize = (256, 256))
        ret, label_img = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY)
        label_img = label_img[np.newaxis, :, :]
        label_img = 255 - label_img
        label_img = label_img/255
        label_img = torch.from_numpy(label_img.astype(np.float32)).clone()

        data = {"input_img": img, "output_img": label_img, "num_node":self.node_cords[i]}
        return data

    def __len__(self):
        return len(self.load_dict)
    
