import torch
from matplotlib import pyplot as plt


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

def plot_node_IoU(result_list):
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

    plt.savefig(f"learnig_prosess/160-200_node_IoU")
    plt.close()