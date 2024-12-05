import torch
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx


def calc_adj_matrix(route):
    n = len(route)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n-1):
        idx_1 = route[i]
        idx_2 = route[i+1]

        adj_matrix[idx_1, idx_2] = 1
        adj_matrix[idx_2, idx_1] = 1
    adj_matrix[idx_2, route[0]] = 1
    adj_matrix[route[0], idx_2] = 1
    return adj_matrix

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


def plot_tsp(p, x_coord, W_val, W_target, title="default"):
    """
    Returns:
        p: Updated figure/subplot
    
    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] == 1:
                    pairs.append((r, c))
        return pairs
    
    G = nx.DiGraph(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    target_pairs = _edges_to_node_pairs(W_target)
    colors = ['g'] + ['b'] * (len(x_coord) - 1)  # Green for 0th node, blue for others
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=target_pairs, alpha=1, width=1, edge_color='r')
    p.set_title(title)
    return p