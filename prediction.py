import numpy as np
import cv2
from skimage.draw import line, disk
import torch
from tqdm import tqdm
import os

from utils.utils import IoU_coef, denormalize, inverse, plot_node_IoU

def compute_edge_pixels(node_cords, edges, img_size=(256, 256), edge_width=3):

    X_list = node_cords.T[0]
    Y_list = node_cords.T[1]
    # 座標をピクセル座標に変換
    X_pixel = (np.array(X_list) * img_size[0]).astype(int)
    Y_pixel = (np.array(Y_list) * img_size[1]).astype(int)

    # エッジのピクセルインデックスを保存する辞書
    edge_pixels = {}

    # 各エッジのピクセル座標を計算
    for start_idx, end_idx in edges:
        # 始点と終点のピクセル座標
        start_point = (Y_pixel[start_idx], X_pixel[start_idx])
        end_point = (Y_pixel[end_idx], X_pixel[end_idx])

        # 線分のピクセル座標を計算 (中央ライン)
        rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])

        # 幅を考慮したピクセル領域を計算
        pixels = []
        for r, c in zip(rr, cc):
            rr_disk, cc_disk = disk((r, c), edge_width // 2, shape=img_size)
            pixels.extend(list(zip(rr_disk, cc_disk)))
        
        # エッジと対応するピクセルインデックスを保存
        edge_pixels[(start_idx, end_idx)] = list(set(pixels))  # 重複を排除

    return edge_pixels

# 使用例
np.random.seed(0)  # 再現性のためのシード設定
num_nodes = 10
X_list = np.random.rand(num_nodes)  # 単位正方形内のx座標
Y_list = np.random.rand(num_nodes)  # 単位正方形内のy座標
edges = [(0, 1), (1, 2), (2, 3)]  # エッジを指定

# エッジピクセルインデックスを計算
edge_pixels = compute_edge_pixels(X_list, Y_list, edges)

# ピクセルインデックスを表示
for edge, pixels in edge_pixels.items():
    print(f"Edge {edge}: {pixels[:5]} ... (total {len(pixels)} pixels)")


class calc_route:
    def __init__(self, img, node_cords):
        self.img = img
        self.node_cords = node_cords
        self.num_node = len(node_cords)

    def update(self, state, agg_type="mean"):
        if state is []:
            state.appand(0)

        crr_node = state[-1]
        edges = [(crr_node, i) for i in range(self.num_node) if i not in state]

         #ピクセル値を求める処理
        
        edge_pixels = compute_edge_pixels(self.node_cords, edges)
        score_dict = {}
        for edge, pixels in edge_pixels.items():
            if agg_type == "mean":
                score_dict[edge] = np.mean([self.img[pixel] for pixel in pixels])
            else:
                score_dict[edge] = (sum(1 for index in pixels if self.img[index] == 0)) / len(pixels)
        sorted_data = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        state.append(sorted_data[0][0][1])
        return state

def predict(model_path, model, test_loader, num_pred_batch, device):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    result_list = []
    for j, data in tqdm(enumerate(test_loader)):  # dataにはバッチでまとまったデータが入る
        if num_pred_batch == j:
            break
        inputs, labels, node_cords = data["img"].to(device), data["label"].to(device), data["node_cords"]
        with torch.no_grad():
            outputs = model(inputs)
            #print(f"{j} batch目")
            for i, (x, y, label, node) in enumerate(zip(inputs, outputs, labels, node_cords)):
                num_node = len(node)

                ins = calc_route(outputs, node_cords)
                route = []
                for i in range(num_node):
                    route = ins.update(route)
                IoU = IoU_coef(label, y)
                result_list.append((IoU, num_node))

                x = x.cpu().detach().clone().numpy()
                x = denormalize(x)
                x = inverse(x)
                x = x.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                y = y.cpu().detach().clone().numpy()
                y = denormalize(y)
                y = inverse(y)
                y = y.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                label = label.cpu().detach().clone().numpy()
                label = denormalize(label)
                label = inverse(label)
                label = label.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                cv2.imwrite("resources/prediction/in_img" + os.sep + f"{num_node}node_input{str(j)}-{str(i)}.jpg", x)
                cv2.imwrite("resources/prediction/pred_img" + os.sep + f"{IoU}_{num_node}node_prediction{str(j)}-{str(i)}.jpg", y)
                cv2.imwrite("resources/prediction/ans_img" + os.sep + f"{num_node}node_ans{str(j)}-{str(i)}.jpg", label)

    plot_node_IoU(result_list)




