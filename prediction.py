import numpy as np
import cv2
from skimage.draw import line, disk
import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import networkx as nx


from utils.utils import IoU_coef, denormalize, inverse, plot_node_IoU, calc_adj_matrix
from utils.model import UNet
from utils.data import TSPDataset, collate_fn




def plot_tsp(p, x_coord, W_val, W_target, title="default"):
    import torch
    def _edges_to_node_pairs(W):
        pairs = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] == 1:
                    pairs.append((r, c))
        return pairs
    
    # Tensorをリストに変換
    if isinstance(x_coord, torch.Tensor):
        x_coord = x_coord.numpy()
    
    G = nx.DiGraph(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))  # ノード位置
    target_pairs = _edges_to_node_pairs(W_target)
    colors = ['g'] + ['b'] * (len(x_coord) - 1)
    
    # ノードとエッジの描画
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, ax=p)
    nx.draw_networkx_edges(G, pos, edgelist=target_pairs, alpha=1, width=1, edge_color='r', ax=p)
    
    # ノード番号（ラベル）を表示
    labels = {i: str(i) for i in range(len(x_coord))}  # ノード番号をラベルとして定義
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=p)  # ラベルを描画
    
    p.set_title(title)
    return p


def compute_edge_pixels(node_cords, edges, img_size=(256, 256), edge_width=4):

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

# # 使用例
# np.random.seed(0)  # 再現性のためのシード設定
# num_nodes = 10
# X_list = np.random.rand(num_nodes)  # 単位正方形内のx座標
# Y_list = np.random.rand(num_nodes)  # 単位正方形内のy座標
# edges = [(0, 1), (1, 2), (2, 3)]  # エッジを指定

# # エッジピクセルインデックスを計算
# edge_pixels = compute_edge_pixels(X_list, Y_list, edges)

# # ピクセルインデックスを表示
# for edge, pixels in edge_pixels.items():
#     print(f"Edge {edge}: {pixels[:5]} ... (total {len(pixels)} pixels)")


class calc_route:
    def __init__(self, img, node_cords):
        self.img = img
        self.node_cords = node_cords
        self.num_node = len(node_cords)
        print("self.img shape is ", self.img.shape)

    def update(self, state, agg_type="mean"):
        if not state:
            state.append(0)

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

        next_node = list(sorted_data.keys())[0]
 
        state.append(next_node[1])
        return state

def predict(model_path, model, test_loader, num_pred_batch):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    result_list = []
    for j, data in tqdm(enumerate(test_loader)):  # dataにはバッチでまとまったデータが入る
        if num_pred_batch == j:
            break
        inputs, labels, node_cords, num_nodes = data["img"].to(device), data["label"].to(device), data["node_cords"], data["num_node"]
        with torch.no_grad():
            outputs = model(inputs)
            #print(f"{j} batch目")
            for i, (x, y, label, node, num_node) in enumerate(zip(inputs, outputs, labels, node_cords, num_nodes)):
                node = node[:int(num_node)]
                np_y = y.squeeze(0).cpu().numpy()
                ins = calc_route(np_y, node)
                route = []
                for i in range(int(num_node)-1):
                    route = ins.update(route)

                print("route is ", route)

                pred_adj_matrix = calc_adj_matrix(route)
                W_val = squareform(pdist(node, metric='euclidean'))
                fig, axes = plt.subplots(1, 3, figsize=(12, 6))

                plot_tsp(axes[0], node, W_val, pred_adj_matrix)

                IoU = IoU_coef(label, y)
                result_list.append((IoU, num_node))

                x = x.cpu().detach().clone().numpy()
                x = denormalize(x)
                x = inverse(x)
                x = x.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                y = y.cpu().detach().clone().numpy()
                y = denormalize(y)
                y = inverse(y)
                y =  np.where(y >= 127, 255, 0)
                y = y.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                label = label.cpu().detach().clone().numpy()
                label = denormalize(label)
                label = inverse(label)
                label = label.reshape((256, 256, 1))  # 画像として保存するためにuint8に変換

                axes[1].imshow(y, cmap="gray")
                axes[1].set_title("U-net output")
                axes[1].axis("off")  # 軸を非表示

                axes[2].imshow(label, cmap="gray") 
                axes[2].set_title("ground trueth image")
                axes[2].axis("off")  # 軸を非表示

                # 全体のレイアウト調整
                plt.tight_layout()

                # 画像の保存
                plt.savefig(f"resources/prediction/sample{i}_node{num_node}.png", dpi=300)

                # プロットの表示
                # plt.show()

                # cv2.imwrite("resources/prediction/in_img" + os.sep + f"{num_node}node_input{str(j)}-{str(i)}.jpg", x)
                # cv2.imwrite("resources/prediction/pred_img" + os.sep + f"{IoU}_{num_node}node_prediction{str(j)}-{str(i)}.jpg", y)
                # cv2.imwrite("resources/prediction/ans_img" + os.sep + f"{num_node}node_ans{str(j)}-{str(i)}.jpg", label)

    plot_node_IoU(result_list)




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device is {device}")

    BATCH_SIZE = 128
    num_epochs = 300
    input_channel_count = 1  # 入力画像のチャンネル数 
    output_channel_count = 1  # 出力画像のチャンネル数
    first_layer_filter_count = 32  # 最初の層のフィルター数
    unet = UNet(input_channel_count, output_channel_count, first_layer_filter_count).to(device)

    loader_args = {"batch_size": BATCH_SIZE, "num_workers": 0}
    test_set = TSPDataset("test", do_convert=True)
    test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate_fn, shuffle=False, **loader_args)

    model_path = 'resources/weight/2024-07-23-1839_bestEPOCH17.pth'
    # model_path = "weight/2024-11-15-0418_epoch20.pth"
    print("num test data is ", len(test_loader))
    predict(model_path, unet, test_loader, num_pred_batch=1)

