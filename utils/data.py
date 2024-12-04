import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import pandas as pd

class TSPDataset(BaseDataset):
    def __init__(self, split, do_convert=False):
        self.do_convert = do_convert
        if self.do_convert:
            self.load_dict = pd.read_pickle(f"resources/{split}_data.pkl")
        else:
            self.load_dict = pd.read_pickle(f"resources/add_convert/{split}_data.pkl")

            
        self.input_img = self.load_dict["input_img"]
        self.output_img = self.load_dict["output_img"]
        self.num_node = self.load_dict["num_node"]
        self.node_cords = self.load_dict["node_cords"]


    def __getitem__(self, i):
        img = self.input_img[i]
        if self.do_convert:
            img = cv2.resize(img, dsize = (256, 256))
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = img[np.newaxis, :, :]
            img = 255 - img
            img = img/255
        img = torch.from_numpy(img.astype(np.float32)).clone()
        #img = img.permute(2, 0, 1) #元々画像(256, 256, 1(チャンネル数))だったものを(1, 256, 256 )のchannel firstに変換

        label_img = self.output_img[i]
        if self.do_convert:
            label_img = cv2.resize(label_img, dsize = (256, 256))
            ret, label_img = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY)
            label_img = label_img[np.newaxis, :, :]
            label_img = 255 - label_img
            label_img = label_img/255
        label_img = torch.from_numpy(label_img.astype(np.float32)).clone()
        #label_img = label_img.permute(2, 0, 1)

        data = {"img": img, "label": label_img, "num_node": self.num_node[i], "node_cords": self.node_cords[i]}
        return data

    def __len__(self):
        return len(self.num_node)