import torch
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import os
import random
import torch.nn.functional as F
import torch.nn as nn


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def IoU_coef(y_true, y_pred):

    #y_pred_sigmoid = torch.sigmoid(y_pred)

    T = y_true.view(-1)
    P = y_pred.view(-1)

    intersection = torch.sum(T * P)
    IoU = (intersection + 1.0) / (torch.sum(T) + torch.sum(P) - intersection + 1.0)
    return IoU.item()

def denormalize(img):
    return img * 255

def normalize(img):
    return img / 255

def inverse(img):
    return 255 - img

def plot_node_IoU(result_list, log_dir):
    sorted_by_node = sorted(result_list, key=lambda x: x[1])
    node_list = [t[1] for t in sorted_by_node]
    IoU_list = [t[0] for t in sorted_by_node]
    print(len(IoU_list))
    print(len(node_list))

    print("IoU->", sum(IoU_list)/len(IoU_list))

    plt.figure()
    plt.plot(node_list, IoU_list)
    plt.xlabel("num_node")
    plt.ylabel("IoU")
    plt.title("node-IoU graph")

    plt.savefig(os.path.join(log_dir, "node-IoU.png"))
    plt.close()


def plot_history(history, logdir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'loss.png'))
    plt.close()
    
    plt.figure()
    plt.plot(epochs, history['train_IoU'], label='Train IoU')
    plt.plot(epochs, history['val_IoU'], label='Val IoU')
    plt.title('IoU')
    plt.xlabel('epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'IoU.png'))
    plt.close()

