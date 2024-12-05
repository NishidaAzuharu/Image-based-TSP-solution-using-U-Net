import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import pandas as pd


class TSPDataset(BaseDataset):
    def __init__(self, split, do_convert=False):
        self.do_convert = do_convert
        self.load_dict = pd.read_pickle(f"resources/node10-150_{split}_data.pkl")
          
        self.input_img = self.load_dict["input_img"]
        self.output_img = self.load_dict["output_img"]
        self.node_cords = self.load_dict["node_cords"]


    def __getitem__(self, i):
        img = self.input_img[i]
        img = cv2.resize(img, dsize = (256, 256))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = img[np.newaxis, :, :]
        img = 255 - img
        img = img/255
        img = torch.from_numpy(img.astype(np.float32)).clone()
        #img = img.permute(2, 0, 1) #元々画像(256, 256, 1(チャンネル数))だったものを(1, 256, 256 )のchannel firstに変換

        label_img = self.output_img[i]
        label_img = cv2.resize(label_img, dsize = (256, 256))
        ret, label_img = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY)
        label_img = label_img[np.newaxis, :, :]
        label_img = 255 - label_img
        label_img = label_img/255
        label_img = torch.from_numpy(label_img.astype(np.float32)).clone()
        #label_img = label_img.permute(2, 0, 1)

        data = {"img": img, "label": label_img, "node_cords": self.node_cords[i], "num_node": len(self.node_cords[i])}
        return data

    def __len__(self):
        return len(self.load_dict)
    

def collate_fn(batch):
    imgs = []
    label_imgs = []
    node_cords = []
    num_nodes = []

    for data in batch:
        img, label_img, node_cord, num_node = data["img"], data["label"], data["node_cords"], data["num_node"]
        imgs.append(torch.tensor(img, dtype=torch.float32))
        label_imgs.append(torch.tensor(label_img, dtype=torch.float32))
        node_cords.append(torch.tensor(node_cord, dtype=torch.float32))
        num_nodes.append(torch.tensor(num_node, dtype=torch.long))


    max_node = max(num_nodes)

    padded_node_cords = []
    for node_cords, n in zip(node_cords, num_nodes):
        paded_cords = torch.zeros((max_node, 2), dtype=torch.float32)
        paded_cords[:n, :] = node_cords
        padded_node_cords.append(paded_cords)

    imgs = torch.stack(imgs, dim=0)
    label_imgs = torch.stack(label_imgs, dim=0)
    node_cords = torch.stack(padded_node_cords, dim=0)
    num_nodes = torch.tensor(num_nodes, dtype=torch.long)

    return {"img": imgs, "label": label_imgs, "node_cords":node_cords, "num_node":num_nodes}

